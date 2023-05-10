#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:2080Ti:1 ####request GPU
#SBATCH -n 1
#SBATCH -t 10-00:00          #### Time before job is ended automatically = 7days.
#SBATCH --job-name=Printing_Hello
#SBATCH --requeue
#SBATCH --mail-type=ALL
#SBATCH --mem=32G                    #### The max amount of memory you expect your job to use at any one time.
                                       ### Probably leave this alone unless the job crashes due to memory issues.

ntimes=$1
python3 'pytorch_training.py'
