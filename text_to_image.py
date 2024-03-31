import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# We enable the memory-efficient attention implementation in PyTorch 2.0 which
# automatically enables several optimizations depending on the inputs and the GPU type.
pipe.unet.set_attn_processor(AttnProcessor2_0())

prompt = "a photo of an astronaut riding a horse on mars"

results = pipe(prompt=prompt, guidance_scale=7.5, num_inference_steps=25)
images = results.images

nsfw_detects = results.nsfw_content_detected
print(nsfw_detects[0])