nvidia-smi | grep 'python' | awk '{print $3}' | xargs -I {} ps -o user= -p {} | sort | uniq
