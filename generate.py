import os
import torch
import argparse
from diffusers import DiffusionPipeline

def generate_images(pipeline, prompts, output_folder, num_images=1):

    os.makedirs(output_folder, exist_ok=True)
    
    for idx, prompt in enumerate(prompts):
        print(f"Generating image for prompt: '{prompt}'...")
        images = pipeline(prompt, num_images_per_prompt=num_images).images
        
        for i, img in enumerate(images):
            img_path = os.path.join(output_folder, f"generated_image_{idx}_{i}.png")
            img.save(img_path)
            print(f"Image saved to {img_path}")

def main(args):
    checkpoint_path = os.path.join(args.checkpoint_folder, "checkpoint-best") 
    unet_path = os.path.join(checkpoint_path, "unet.pt")
    text_encoder_path = os.path.join(checkpoint_path, "text_encoder.pt")

    torch_dtype = torch.bfloat16
    
    pretrained_model_name_or_path = "stablediffusionapi/cyberrealistic-41"
    
    print(f"Loading pretrained model: {pretrained_model_name_or_path}")
    pipeline = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,  
        safety_checker=None,
    )
    
    print(f"Loading fine-tuned weights from {unet_path} and {text_encoder_path}")
    pipeline.unet = torch.load(unet_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    pipeline.text_encoder = torch.load(text_encoder_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.prompts_file, "r") as f:
        prompts = [line.strip() for line in f.readlines()]

    generate_images(pipeline, prompts, args.output_folder, num_images=args.num_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using fine-tuned LoRA Stable Diffusion model")
    parser.add_argument("--checkpoint_folder", type=str, required=True, help="Folder containing the fine-tuned model checkpoints")
    parser.add_argument("--prompts_file", type=str, required=True, help="File containing prompts for image generation")
    parser.add_argument("--output_folder", type=str, default="./generated_images", help="Folder to save generated images")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate per prompt")
    args = parser.parse_args()

    main(args)

