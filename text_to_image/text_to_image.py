import torch
import argparse
import time
from accelerate import PartialState
from diffusers import StableDiffusionPipeline

def read_prompts_from_file(file_path):
    """read more prompts from file stream"""

    with open(file_path, 'r') as file:
        prompts = [line.strip() for line in file.readlines()]
    return prompts

def parse_args():
    """ adapt for future new features"""

    parser = argparse.ArgumentParser(description='inference args')
    parser.add_argument('--dist_inference', type=bool, default=False, help='enable multi gpus inferencc')
    parser.add_argument('--img_save_dir', type=str, default='output', help='inference images save directory')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    start_time = time.time()
    prompts = read_prompts_from_file('prompts.txt')
    if args.dist_inference:
        print(f'dist inference:')
        dist_state = PartialState()
        pipe.to(dist_state)
        with dist_state.split_between_processes(prompts) as prompt:
            result = pipe(prompt).images[0]
            result.save(f'{args.img_save_dir}/result_{dist_state.process_index}_prompt.png')
    else:
        print(f'singel gpu inference:')
        pipe.to('cuda:0')
        for i, prompt in enumerate(prompts):
            print(f'the {i} start process')
            result = pipe(prompt=prompt).images[0]
            result.save(f'{args.img_save_dir}/result_{i}_prompt.png')
            print(f'the {i} prompt findished process')

    end_time = time.time()
    execute_cost = end_time - start_time
    print(f'execute cost time: {execute_cost} sec')

if __name__ == '__main__':
    main()
