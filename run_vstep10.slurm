#!/bin/bash

#SBATCH --job-name=test
#SBATCH --time=48:00:00
##SBATCH --cpus-per-task=1
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

cd /flush5/sho092/Robust_learning/MAML-Pytorch


python  sr_train.py --num_workers 1 --data_dist_same False --crop_size 120 --update_step 10 

