#!/bin/bash
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1 # number of tasks per node
#SBATCH -t 5-23:59 # Runtime in D-HH:MM
#SBATCH -p cpu # gpu partition

#SBATCH -o hostname_%j.out
#SBATCH -e hostname_%j.err

# Load required package to run your job.
#module load python/3.6.3 
module load gcc/7.2.0
module load anaconda3/5.0.1

#python -u train_low.py --checkpoint_foldername 0 
python -u train_low.py --checkpoint_foldername 2 --continue_training True
