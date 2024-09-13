

## Introduction

This repository provides a method to finetune the text-to-image model, Stable Diffusion, into a personalized model capable of consistently generating images of a specific individual. The fine-tuning process leverages Hugging Face's PEFT LoRA technique.

After finetuning with Brad Pitt's images, the model can generate new images that consistently resemble Brad Pitt based on text prompts.

Additionally, metrics such as face distance and CLIP score are evaluated for the finetuned model.

Below is an illustration of images generated after 1000 steps of finetuning.

<img src=".\images\generated.png" alt="generated" />

## Install packages

```bash
pip install timm fairscale transformers requests accelerate diffusers einop safetensors voluptuous jax peft deepface tensorflow keras
```

## Finetune  Stable Diffusion with Hugging Face's PEFT LoRA

```
python finetune.py --data_folder ./datasets/brad --prompts_folder ./datasets/prompts 
```

## Generate images using finetuned Stable Diffusion

```bash
python generate.py --checkpoint_folder ./logs --prompts_file ./datasets/prompts/validation_prompt.txt --output_folder ./generated_images --num_images 1
```

## Evaluate metrics using finetuned Stable Diffusion

```
python evaluate.py --checkpoint_folder ./logs --images_folder ./datasets/brad --prompts_file ./datasets/prompts/validation_prompt.txt --output_folder ./inference 
```

