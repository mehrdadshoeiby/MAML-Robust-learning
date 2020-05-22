#!/bin/bash

#SBATCH --job-name=test
#SBATCH --time=12:00:00
##SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sho092@csiro.au
#SBATCH --mem=64gb
#bracewell.hpc.csiro.au

module load slurm
  
##module load pytorch/0.4.1-py36-cuda91
##module load torchvision/0.2.1-py36

pwd

cd /flush5/sho092/MAML_Pytorch

python  sr_train.py --meta_lr 0.0001 --update_lr 0.001 --epoch 10000
