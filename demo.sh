#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=200GB
#SBATCH --time=12:00:00
#SBATCH --array=0
#SBATCH --job-name=glide_demo
#SBATCH --output=glide_demo_%A_%a.out

module purge
module load cuda-11.4

python -u /misc/vlgscratch4/LakeGroup/emin/glide-text2im/demo.py

echo "Done"
