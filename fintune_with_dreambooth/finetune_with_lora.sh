#!/bin/bash

source ../env.sh

export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
export OUTPUT_DIR="output/${SLURM_JOB_ID}"

echo "Time to finetune:"
begin_time=`date +'%Y-%m-%d %H:%M:%S'`

accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --caption_column="text" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=10 \
  --checkpointing_steps=500 \
  --gradient_checkpointing \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --validation_prompt="cute dragon creature" \
  --output_dir=$OUTPUT_DIR


echo "Finished finetune"
end_time=`date +'%Y-%m-%d %H:%M:%S'`

# calculat cost time
start_seconds=$(date --date="$begin_time" +%s);
end_seconds=$(date --date="$end_time" +%s);
echo "Cost time "$((end_seconds-start_seconds))"s"
