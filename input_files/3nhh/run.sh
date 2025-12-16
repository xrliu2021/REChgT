#!/bin/bash
  
export fname=rest-ex2d-msld-vbias-protonation-protein
export nchrg=4
export npH=6
export np=$(($npH * $nchrg))

export comp=3nhh

mkdir -p log

# submit jobs
sbatch --job-name=${comp} --ntasks=$np $DEPEND ./run_rest_ex2d.slurm

