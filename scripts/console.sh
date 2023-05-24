#!/bin/bash

set e

# Check if the current directory is named "scripts"
current_directory=$(basename "$PWD")

if [ "$current_directory" == "scripts" ]; then
  echo "Error: This script should be called from the root of the project."
  echo "Example: bash ./scripts/console.sh"
  exit 1
fi

docker run -it \
    --user $(id -u):$(id -g) \
    --gpus='device=0' \
    --ipc host \
    --env WANDB_API_KEY \
    -v "/scratch1/ozimmermann/cache:/home/mamba/.cache" \
    -v "$(pwd)":/workspace \
    -w /workspace \
    nlp_template \
    bash

# the mounted cache folder has to exist somewhere, befor this script can be run