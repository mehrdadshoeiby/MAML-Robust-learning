#!/bin/bash

#SBATCH --job-name=test
#SBATCH --time=24:00:00
##SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sho092@csiro.au
#SBATCH --mem=64gb
#bracewell.hpc.csiro.au

module load slurm
module load torchvision/0.4.0-py36 

##module load pytorch/0.4.1-py36-cuda91
##module load torchvision/0.2.1-py36

pwd

python  sr_train.py --num_workers 1 --k 4 --update_step 5 

