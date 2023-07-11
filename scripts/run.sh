#!/bin/bash

set e

# Check if the current directory is named "scripts"
current_directory=$(basename "$PWD")

if [ "$current_directory" == "scripts" ]; then
  echo "Error: This script should be called from the root of the project."
  echo "Example: bash ./scripts/console.sh"
  exit 1
fi

python train.py --wandb_run_name="firstTestInGroup" --val_before_training=False --training_goal=1000 -d ./data/sw --batch_size_per_device=1 --force_deterministic=True