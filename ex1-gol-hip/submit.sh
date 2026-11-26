#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=courses0100-gpu
#SBATCH --reservation=CurtinHPCcourse2026-gpu
#SBATCH --partition=gpu
#SBATCH --time=00:30:00

module load rocm/6.4.1

sizes=(
    "10 10"
    "100 100"
    "500 500"
    "1000 1000"
    "2500 2500"
    "5000 5000"
    "7500 7500"
    "10000 10000"
    "20000 20000"
    "30000 30000"
    "1000 10000"
)

for size in "${sizes[@]}"; do
    echo ""
    echo "Size: $size"
    ./bin/game_of_life_hip $size 100
    echo ""
done

