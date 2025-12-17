# Input files for 2D REChgT simulation of 3NHH

Reference: 
Enhanced Sampling of Buried Charges in Free Energy Calculations Using Replica Exchange with Charge Tempering
Xiaorong Liu and Charles L. Brooks, III
J. Chem. Theory Comput. 2024, 20, 3, 1051–1061

## File description

toppar_c36_jul21/: CHARMM36m force field related topology and parameter files

input_files/: Input files and scripts to run 2D REChgT simulation of 3NHH

## Simulation software

pyCHARMM: It needs to be compiled with BLaDE, skip mpi

## How to run this example simulation?

```shell
# activate the Python environment where pyCHARMM is installed, for example,
conda activate pyCHARMM

cd input_files/3nhh/
sh run.sh
```

Note that in this example, SLURM is used. Therefore, it's necessary to modify 
file run_rest_ex2d.slurm in input_files/3nhh/ to request appropriate resources
(e.g., partition, number of GPUs per node). 

Also, we may need to pay attention to the "mpirun" command in run_rest_ex2d.slurm (line 56).
Depending on the version of mpirun, the syntax may need to be slightly modified. 
