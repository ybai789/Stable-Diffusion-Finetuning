import os
import glob
import torch
import argparse
import cv2
import numpy as np

from deepface import DeepFace
from transformers import AutoProcessor, AutoModel
from diffusers import DiffusionPipeline

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

def evaluate(pretrained_model_name_or_path, weight_dtype, seed, unet_path, text_encoder_path, validation_prompt, output_folder, train_emb):
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    pipeline.unet = torch.load(unet_path)
    pipeline.text_encoder = torch.load(text_encoder_path)
    pipeline = pipeline.to(DEVICE)
    
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = AutoModel.from_pretrained(clip_model_name)
    clip_processor = AutoProcessor.from_pretrained(clip_model_name)

    with torch.no_grad():
        generator = torch.Generator(device=DEVICE)
        generator = generator.manual_seed(seed)
        face_score = 0
        clip_score = 0
        mis = 0
        print("Generating validation images...")
        images = pipeline(validation_prompt, num_inference_steps=30, generator=generator).images
        valid_emb = []
        for i, image in enumerate(images):
            save_file = f"{output_folder}/valid_image_{i}.png"
            image.save(save_file)
            opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            emb = DeepFace.represent(
                opencvImage,  
                detector_backend="retinaface",
                model_name="GhostFaceNet",
                enforce_detection=False,
            )

            if emb == [] or emb[0]['face_confidence'] == 0:
                mis += 1
                continue
            emb = emb[0]
            inputs = clip_processor(text=validation_prompt[i], images=image, return_tensors="pt")
            outputs = clip_model(**inputs)
            sim = outputs.logits_per_image
            clip_score += sim.item()
            valid_emb.append(emb['embedding'])
        if len(valid_emb) == 0:
            return 0, 0, mis
        valid_emb = torch.tensor(valid_emb)
        valid_emb = (valid_emb / torch.norm(valid_emb, p=2, dim=-1)[:, None]).cuda()
        train_emb = (train_emb / torch.norm(train_emb, p=2, dim=-1)[:, None]).cuda()
        face_score = torch.cdist(valid_emb, train_emb, p=2).mean().item()
        clip_score /= len(validation_prompt) - mis
    return face_score, clip_score, mis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LoRA fine-tuned Stable Diffusion model")
    parser.add_argument("--checkpoint_folder", type=str, required=True, help="Folder containing the model checkpoints")
    parser.add_argument("--images_folder", type=str, required=True, help="Folder containing training images")
    parser.add_argument("--prompts_file", type=str, required=True, help="File containing validation prompts")
    parser.add_argument("--output_folder", type=str, default="./inference", help="Folder to save inference results")
    args = parser.parse_args()

    pretrained_model_name_or_path = "stablediffusionapi/cyberrealistic-41"
    seed = 1126
    weight_dtype = torch.bfloat16

    checkpoint_path = os.path.join(args.checkpoint_folder, "checkpoint-best") 
    unet_path = os.path.join(checkpoint_path, "unet.pt")
    text_encoder_path = os.path.join(checkpoint_path, "text_encoder.pt")

    inference_path = args.output_folder
    os.makedirs(inference_path, exist_ok=True)

    train_image_paths = []
    for ext in IMAGE_EXTENSIONS:
        train_image_paths.extend(glob.glob(f"{args.images_folder}/*{ext}"))
    train_image_paths = sorted(train_image_paths)
    
    train_emb = torch.tensor([DeepFace.represent(img_path, detector_backend="retinaface", model_name="GhostFaceNet", enforce_detection=False)[0]['embedding'] for img_path in train_image_paths])

    with open(args.prompts_file, "r") as f:
        validation_prompt = [line.strip() for line in f.readlines()]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    face_score, clip_score, mis = evaluate(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        weight_dtype=weight_dtype,
        seed=seed,
        unet_path=unet_path,
        text_encoder_path=text_encoder_path,
        validation_prompt=validation_prompt,
        output_folder=inference_path,
        train_emb=train_emb,
    )
    
    print("Face Similarity Score:", face_score, "CLIP Score:", clip_score, "Faceless Images:", mis)

