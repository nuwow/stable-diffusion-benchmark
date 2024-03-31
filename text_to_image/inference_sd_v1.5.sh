#!/bin/bash

# env
source  ../env.sh


# python text_to_image.py
accelerate launch --num_processes=2 text_to_image.py
