import argparse
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from peft import get_peft_model, LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from deepface import DeepFace
import cv2
from torch.utils.data import Dataset
from evaluate import evaluate
from transformers import get_scheduler


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

class Text2ImageDataset(Dataset):
    def __init__(self, folder, transform, tokenizer):
        self.image_paths = []
        self.caption_paths = []
        
        for file_name in sorted(os.listdir(folder)):
            file_path = os.path.join(folder, file_name)
            if any(file_name.endswith(ext) for ext in IMAGE_EXTENSIONS):
                self.image_paths.append(file_path)
            elif file_name.endswith('.txt'):
                self.caption_paths.append(file_path)
        
        captions = []
        for path in self.caption_paths:
            with open(path, "r", encoding="utf-8") as f:
                captions.append(f.readline().strip())
        
        self.inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        if len(self.image_paths) != len(self.inputs.input_ids):
            print(f"Warning: Number of images ({len(self.image_paths)}) and captions ({len(self.inputs.input_ids)}) do not match.")
            min_length = min(len(self.image_paths), len(self.inputs.input_ids))
            self.image_paths = self.image_paths[:min_length]
            self.inputs.input_ids = self.inputs.input_ids[:min_length]

        self.transform = transform
        self.train_emb = self.compute_train_embeddings()

    def compute_train_embeddings(self):
        print("Computing face embeddings for training images...")
        embeddings = []
        for img_path in self.image_paths:
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                emb = DeepFace.represent(
                    img,
                    detector_backend="retinaface",
                    model_name="GhostFaceNet",
                    enforce_detection=False,
                )
                
                if emb and emb[0]['face_confidence'] > 0:
                    embeddings.append(emb[0]['embedding'])
                else:
                    print(f"Warning: No face detected in {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        
        if not embeddings:
            print("Warning: No valid face embeddings found in the dataset.")
            return torch.tensor([])
        
        return torch.tensor(embeddings)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        input_id = self.inputs.input_ids[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            tensor = self.transform(image)
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")
            return None
        
        return tensor, input_id

    def __len__(self):
        return len(self.image_paths)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def prepare_lora_model(pretrained_model_name_or_path="stablediffusionapi/cyberrealistic-41"):
    model_path = os.path.join(os.getcwd(), "models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    text_encoder_path = os.path.join(model_path, "text_encoder.pt")
    if not os.path.exists(text_encoder_path):
        print("Downloading text_encoder...")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        torch.save(text_encoder, text_encoder_path)
    else:
        text_encoder = torch.load(text_encoder_path)

    unet_path = os.path.join(model_path, "unet.pt")
    if not os.path.exists(unet_path):
        print("Downloading unet...")
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
        torch.save(unet, unet_path)
    else:
        unet = torch.load(unet_path)

    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    lora_rank = 32
    lora_alpha = 16
    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )

    text_encoder_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    unet = get_peft_model(unet, unet_lora_config)
    text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)

    vae.requires_grad_(False)

    unet.to(DEVICE, dtype=torch.bfloat16)
    vae.to(DEVICE, dtype=torch.bfloat16)
    text_encoder.to(DEVICE, dtype=torch.bfloat16)

    return text_encoder, unet, vae, noise_scheduler


def collate_fn(examples):
    pixel_values = []
    input_ids = []
    
    for tensor, input_id in examples:
        if tensor is not None:  
            pixel_values.append(tensor)
            input_ids.append(input_id)

    pixel_values = torch.stack(pixel_values, dim=0).float()
    input_ids = torch.stack(input_ids, dim=0)

    return {"pixel_values": pixel_values, "input_ids": input_ids}
    
def main(args):
    project_dir = os.getcwd()
    print(f"Using current directory as project_dir: {project_dir}")

    output_folder = os.path.join(project_dir, "logs") 
    seed = 1126 
    train_batch_size = 2 
    resolution = 512 
    weight_dtype = torch.bfloat16 

    pretrained_model_name_or_path = "stablediffusionapi/cyberrealistic-41"
    learning_rate = 1e-4 
    lr_scheduler_name = "cosine_with_restarts" 
    lr_warmup_steps = 100 

    max_train_steps = 1000 
    validation_prompt = "validation_prompt.txt"
    validation_prompt_path = os.path.join(args.prompts_folder, validation_prompt)
    validation_prompt_num = 3 
    validation_step_ratio = 0.2 
    with open(validation_prompt_path, "r") as f:
        validation_prompt = [line.strip() for line in f.readlines()]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose(
        [
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    dataset = Text2ImageDataset(
        folder=args.data_folder,
        transform=train_transform,
        tokenizer=tokenizer
    )

    if dataset.train_emb.numel() == 0:
        print("Error: No valid face embeddings found. Please check your training images.")
        return
    
    text_encoder, unet, vae, noise_scheduler = prepare_lora_model(pretrained_model_name_or_path="stablediffusionapi/cyberrealistic-41")

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,  
        batch_size=train_batch_size,
        num_workers=4,
    )

    print("Dataset and model preparation finished!")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=0,
        desc="Steps",
    )
    global_step = 0
    num_epochs = math.ceil(max_train_steps / len(train_dataloader))
    validation_step = int(max_train_steps * validation_step_ratio)
    best_face_score = float("inf")

    params_to_optimize = (
        list(filter(lambda p: p.requires_grad, unet.parameters())) +
        list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
    )
    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)

    lr_scheduler = get_scheduler(
        name=lr_scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    for epoch in range(num_epochs):
        unet.train()
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            if global_step >= max_train_steps:
                break

            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(DEVICE, dtype=weight_dtype)).latent_dist.sample()
        
            latents = latents * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(batch["input_ids"].to(DEVICE), return_dict=False)[0]
            target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            global_step += 1

            if global_step % validation_step == 0 or global_step == max_train_steps:
                save_path = os.path.join(output_folder, f"checkpoint-last")
                unet_path = os.path.join(save_path, "unet.pt")
                text_encoder_path = os.path.join(save_path, "text_encoder.pt")
                os.makedirs(save_path, exist_ok=True)
                torch.save(unet, unet_path)
                torch.save(text_encoder, text_encoder_path)
                face_score, clip_score, mis = evaluate(
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    weight_dtype=weight_dtype,
                    seed=seed,
                    unet_path=unet_path,
                    text_encoder_path=text_encoder_path,
                    validation_prompt=validation_prompt[:validation_prompt_num],
                    output_folder=save_path,
                    train_emb=dataset.train_emb
                )
                print(f"Step: {global_step}, Face Similarity Score: {face_score}, CLIP Score: {clip_score}, Faceless Images: {mis}")
                if face_score < best_face_score:
                    best_face_score = face_score
                    save_path = os.path.join(output_folder, "checkpoint-best")
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(unet, os.path.join(save_path, "unet.pt"))
                    torch.save(text_encoder, os.path.join(save_path, "text_encoder.pt"))

    print("Fine-tuning Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Stable Diffusion model with LoRA.")
    parser.add_argument("--data_folder", type=str, required=True, help="Folder for both images and captions")
    parser.add_argument("--prompts_folder", type=str, required=True, help="Folder for prompts")
    args = parser.parse_args()

    main(args)

