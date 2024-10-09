import os
import re
import torch
import argparse
import time
import datetime
import random


def get_gpu_free_memory():
    # Use nvidia-smi command to get GPU memory information
    result = os.popen("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits").read()
    free_memory = [int(x) for x in result.strip().split("\n")]
    return free_memory


def find_available_gpus(min_memory, num_gpus):
    free_memory = get_gpu_free_memory()
    available_gpus = [i for i, mem in enumerate(free_memory) if mem >= min_memory]
    if len(available_gpus) >= num_gpus:
        return available_gpus[:num_gpus]
    return None


def allocate_tensor_on_gpus(gpu_ids, min_memory):
    if gpu_ids is None:
        print("Not enough GPUs with sufficient memory available.")
        return
    
    # Allocate tensor on the specified GPUs with specified memory
    for gpu_id in gpu_ids:
        device = torch.device(f"cuda:{gpu_id}")
        random_increment = random.randint(1, 10)
        adjusted_memory = min_memory + random_increment
        tensor_size = (adjusted_memory, 256, 1024)  # Adjust tensor size based on memory to allocate
        tensor = torch.randn(tensor_size, device=device, dtype=torch.float32)
        print(f"Allocated tensor of size {tensor_size} on GPU {gpu_id}")


def main():
    # Use argparse to get command line arguments
    parser = argparse.ArgumentParser(description="GPU Memory Allocation Script")
    parser.add_argument("-m", "--min_memory", type=int, required=True, help="Minimum memory (in MB) required per GPU")
    parser.add_argument("-n","--num_gpus", type=int, required=True, help="Number of GPUs to use")
    args = parser.parse_args()
    
    # Find the specified number of GPUs with at least the required memory
    start_time = datetime.datetime.now()
    while True:
        gpu_ids = find_available_gpus(min_memory=args.min_memory, num_gpus=args.num_gpus)
        elapsed_time = datetime.datetime.now() - start_time
        print(rf"Find {gpu_ids} GPUs have {args.min_memory} MiB memory after {elapsed_time}.")
        if gpu_ids is not None:
            break
        time.sleep(60)  # Wait for 1 minute before trying again
    
    # If suitable GPUs are found, allocate tensor on them
    allocate_tensor_on_gpus(gpu_ids, args.min_memory)
    allocation_time = datetime.datetime.now()
    
    while True:
        print(f"Successfully allocated memory on the {gpu_ids} GPUs with {args.min_memory} MiB memory. Allocation took {datetime.datetime.now() - allocation_time}.")
        print("Press Ctrl+C to stop the process.")
        time.sleep(5)


if __name__ == "__main__":
    main()
