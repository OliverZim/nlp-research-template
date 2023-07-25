#!/bin/bash

set e

# Check if the current directory is named "scripts"
current_directory=$(basename "$PWD")

if [ "$current_directory" == "scripts" ]; then
  echo "Error: This script should be called from the root of the project."
  echo "Example: bash ./scripts/console.sh"
  exit 1
fi

# python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="workers0" --workers=0 #failed 
python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="workers1" --workers=1
python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="workers2" --workers=2
python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="workers4" --workers=4  #default
python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="workers8" --workers=8
python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="workers16" --workers=16

python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="precision16" --precision=16-mixed
python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="precision16mixed" --precision=bf16-mixed
python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="precision32" --precision=32  #default

python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="compileFalse" --compile=False #default
python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="compileTrue" --compile=True

python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="batchSize20" --batch_size_per_device=20
python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="batchSize16" --batch_size_per_device=16
python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="batchSize8" --batch_size_per_device=8 #default
python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="batchSize4" --batch_size_per_device=4

python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="DeterministicNo" --force_deterministic=False #default
python train.py --val_before_training=False --training_goal=40000 -d ./data/sw --wandb_run_name="DeterministicYes" --force_deterministic=True

