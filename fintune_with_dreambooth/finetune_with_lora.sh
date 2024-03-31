#!/bin/bash

source ../env.sh

export MODEL_NAME='runwayml/stable-diffusion-v1-5'
export DATASET_NAME='lambdalabs/pokemon-blip-captions'
export OUTPUT_DIR='output/${SLURM_JOB_ID}'

echo "Time to finetune:"
begin_time=`date +'%Y-%m-%d %H:%M:%S'`


acclerate launch --mixed_precision=fp16' train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 \
  --center_crop \ 
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR


echo "Finished finetune"
end_time=`date +'%Y-%m-%d %H:%M:%S'

# calculat cost time
start_seconds=$(date --date="$begin_time" +%s);
end_seconds=$(date --date="$end_time" +%s);
echo "Cost time "$((end_seconds-start_seconds))"s"
