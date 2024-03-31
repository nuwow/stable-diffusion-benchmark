import torch
import argparse
import time
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# We enable the memory-efficient attention implementation in PyTorch 2.0 which
# automatically enables several optimizations depending on the inputs and the GPU type.
pipe.unet.set_attn_processor(AttnProcessor2_0())

prompts = ["a photo of an astronaut riding a horse on mars",
           'a beautiful sunset over a calm lake',
           'a cat playing with a ball of yarn',
           'A beautiful and powerful mysterious sorceress, smile, sitting on a rock, lightning magic, hat, detailed leather clothing with gemstones, dress, castle background, digital art, hyperrealistic, fantasy, dark art, artstation, highly detailed, sharp focus, sci-fi, dystopian, iridescent gold, studio lighting',
           'A full body Batman character, realistic, ultra detailing, nice dynamic pose, 8K sharp focus, highly detailed, photorealism, armored luxury suit with white and gold chrome details and matte black',
           'A beautiful painting of water spilling out of a broken pot, earth colored clay pot, vibrant background, by greg rutkowski and thomas kinkade, Trending on artstation, hyperrealistic, extremely detailed'
           ]

batch_results = []

start_time = time.time()

for i, prompt in enumerate(prompts):
    print(f'the {i} start process')
    results = pipe(prompt=prompt, guidance_scale=7.5, num_inference_steps=25)
    batch_results.append(results)
    print(f'the {i} prompt findished process')

end_time = time.time()
execute_cost = end_time - start_time
print(f'execute cost time: {execute_cost} sec')
