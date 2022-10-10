import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
token = "<token>"
model_id = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionPipeline.from_pretrained(model_id, 
        revision="fp16", torch_dtype=torch.float16, use_auth_token=token)

pipe.to("cuda")

prompt = "Beautiful Landscape with mountains, inspired by Firewatch artwork, sunset"

with autocast("cuda"):
    output = pipe(prompt)
    image = output["sample"][0]
    file = prompt.replace(" ", "_").replace(",", "")
    image.save(f"{file}.png")
