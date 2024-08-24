nvidia-smi | grep 'python' | awk '{print $5}' | xargs -I {} ps -o user= -p {} | sort -u
