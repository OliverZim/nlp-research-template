#!/bin/bash

set e

# Check if the current directory is named "scripts"
current_directory=$(basename "$PWD")

if [ "$current_directory" == "scripts" ]; then
  echo "Error: This script should be called from the root of the project."
  echo "Example: bash ./scripts/console.sh"
  exit 1
fi

python train.py -d ./data/loremIpsum --wandb_run_name="workers0" --workers=0
python train.py -d ./data/loremIpsum --wandb_run_name="workers1" --workers=1
python train.py -d ./data/loremIpsum --wandb_run_name="workers2" --workers=2
python train.py -d ./data/loremIpsum --wandb_run_name="workers4" --workers=4
python train.py -d ./data/loremIpsum --wandb_run_name="workers8" --workers=8
python train.py -d ./data/loremIpsum --wandb_run_name="workers16" --workers=16

python train.py -d ./data/loremIpsum --wandb_run_name="precision16" --precision=16-mixed
python train.py -d ./data/loremIpsum --wandb_run_name="precision16mixed" --precision=bf16-mixed
python train.py -d ./data/loremIpsum --wandb_run_name="precision32" --precision=32

python train.py -d ./data/loremIpsum --wandb_run_name="compileFalse" --compile=False
python train.py -d ./data/loremIpsum --wandb_run_name="compileTrue" --compile=True

python train.py -d ./data/loremIpsum --wandb_run_name="batchSize" --batch_size_per_device=
python train.py -d ./data/loremIpsum --wandb_run_name="batchSize" --batch_size_per_device=
python train.py -d ./data/loremIpsum --wandb_run_name="batchSize" --batch_size_per_device=

python train.py -d ./data/loremIpsum --wandb_run_name="DeterministicNo" --force_deterministic=False
python train.py -d ./data/loremIpsum --wandb_run_name="DeterministicYes" --force_deterministic=True

