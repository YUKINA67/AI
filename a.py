import torch
from diffusers import StableDiffusionPipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text = int(input("ぷろんぷと"))

pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', revision='fp16')
pipe = pipe.to(device)
generator = torch.Generator().manual_seed(42)
image = pipe(text, guidance_scale=7.5, generator=generator).images[0]
image.save(f'{text}.png')
print("画像を保存しました。")