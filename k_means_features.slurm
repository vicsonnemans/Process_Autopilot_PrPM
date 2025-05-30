#!/bin/bash
#SBATCH --job-name=rl_array
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-15:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/abstraction_%A_%a.err    

module purge
module load Anaconda3/2024.02-1

# Initialize conda in batch job
source ~/.bashrc
conda activate rl_env

# Define configs as lists (same order)
ABSTRACTIONS=("k_means_features" "k_means_features" "k_means_features" "k_means_features" "k_means_features")
KS=(10 50 200 300 500)  # 0 if k not used (e.g., for False)

# Fetch the current parameter values
ABSTRACTION=${ABSTRACTIONS[$SLURM_ARRAY_TASK_ID]}
K=${KS[$SLURM_ARRAY_TASK_ID]}

# Run the Python script with all arguments
python3 simulated_offline_online_RL.py $ABSTRACTION $K 

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s


