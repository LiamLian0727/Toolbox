#!/bin/bash

# Enable debug mode to print each command and its arguments as they are executed.
set -x

# Define the threshold for memory usage percentage and the number of GPUs required.
THRESHOLD=${1:-100}
NUMGPU=${2:-2}

# Validate the input to ensure they are numbers and not negative.
if ! [[ "$THRESHOLD" =~ ^[0-9]+$ ]] || [ "$THRESHOLD" -lt 0 ] || ! [[ "$NUMGPU" =~ ^[0-9]+$ ]] || [ "$NUMGPU" -lt 1 ]; then
    echo "Invalid input. Please enter positive integers for threshold and number of GPUs."
    exit 1
fi

# Infinite loop to keep checking for GPUs until the required number is found and the command is executed.
while true; do
    # Retrieve the status of all GPUs.
    gpus=$(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits)

    # Check if any GPUs are detected.
    if [ -z "$gpus" ]; then
        echo "No NVIDIA GPUs detected or nvidia-smi command is not available. Retrying in 10 seconds..."
        sleep 10
        continue # Go back to the start of the loop.
    fi

    # Initialize an array to hold the indices of available GPUs.
    available_gpus=()

    # Process each GPU and check for memory usage less than the threshold.
    while IFS=',' read -r index free_memory total_memory; do
        # Calculate the used memory and memory usage percentage.
        used_memory=$((total_memory - free_memory))
        usage_percentage=$((used_memory * 100 / total_memory))

        # Check if the memory usage is less than the threshold.
        if [ "$usage_percentage" -lt "$THRESHOLD" ]; then
            echo "GPU Index: $index, Memory Usage: ${usage_percentage}%, Free Memory: ${free_memory} MiB, Total Memory: ${total_memory} MiB"
            available_gpus+=("$index")
        fi
    done <<< "$gpus"

    # Check if the required number of suitable GPUs is available.
    if [ ${#available_gpus[@]} -ge "$NUMGPU" ]; then
        echo "Found sufficient GPUs. Running PyTorch command..."

        # Set the CUDA_VISIBLE_DEVICES environment variable.
        gpu_list=$(IFS=,; echo "${available_gpus[*]}")
        export CUDA_VISIBLE_DEVICES=$gpu_list
        
        # Run the PyTorch command

        # ------------------------------------------ PyTorch Command Block ------------------------------------------.
        cd /root/UIIS-Net
        bash tools/dist_train.sh project/SAM2/config/sam2_baseplus_l_prompt_bbox_mask_decoder_amp_2x_uiis10k.py 2
        # ------------------------------------------ PyTorch Command Block ------------------------------------------.

        # If the PyTorch command is successful, break out of the loop.
        if [ $? -eq 0 ]; then
            echo "PyTorch command executed successfully. Exiting."
            break
        else
            echo "PyTorch command failed. Retrying in 10 seconds..."
            sleep 10
            continue # Go back to the start of the loop.
        fi
    else
        echo "Less than $NUMGPU GPUs with memory usage less than $THRESHOLD% were found. Retrying in 10 seconds..."
        sleep 10
        continue # Go back to the start of the loop.
    fi
done
