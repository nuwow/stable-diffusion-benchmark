#!/bin/bash

# env
source  ../env.sh


# python text_to_image.py
#accelerate launch text_to_image.py
accelerate launch --num_processes=4 text_to_image.py --dist_inference True
