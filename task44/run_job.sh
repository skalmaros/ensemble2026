#!/bin/bash
#SBATCH --job-name=task4
#SBATCH --partition=tutorial
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/out_%j.log
#SBATCH --time=04:00:00

module add GCCcore/13.2.0 Python/3.11.5
source .venv/bin/activate

python pipline.py