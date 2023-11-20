#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --export=ALL
#SBATCH --array=1-100%1
#SBATCH --no-requeue
#SBATCH -o log/ex2d-%A_%a.out

charmm=/home/xrliu/program/charmm/charmm-20220908-blade-rest/build_blade_nompi/bin/bin/charmm
ncycles=500

nitt=1
ii=$SLURM_ARRAY_TASK_ID
ibeg=$(( ( $ii - 1 ) * $nitt + 1 ))
iend=$(( $ii * $nitt ))

#source ./sourceme.sh 
set -x
srun hostname
set +x

DIR=`pwd`

function check_complete(){
   skip=0
   if [[ -f cond0/dcd/${comp}_prod$itt.dcd ]]; then
      nf=`catdcd cond0/dcd/${comp}_prod$itt.dcd  | grep 'Total frames' | awk '{print $3}'`
      if [ $nf -eq $ncycles ]; then
         dcd_size=`ls -l cond0/dcd/${comp}_prod$itt.dcd | awk '{print $5}'`
         ndcd=`ls -l cond*/dcd/${comp}_prod$itt.dcd | grep $dcd_size | wc -l | awk '{print $1}'`
         if [ $ndcd -eq $np ]; then
            skip=1
         fi
      fi
   fi
}


for itt in `seq $ibeg $iend`
do
     finish=0
     if [ -f exchange-all-$itt.dat ]; then
        exch_size=`wc -l exchange-all-$itt.dat | awk '{print $1}'`
        nlines=`echo $ncycles $np | awk '{print $1*(1+$2)+1}'`
        if [ $exch_size -eq $nlines ]; then
           finish=1
        fi
     fi

     if [ $finish -eq 0 ]; then
        echo "$comp $itt"
        mpirun -np $np --quiet -x OMP_NUM_THREADS=1 python $fname.py $comp $itt $SLURM_NTASKS_PER_NODE
     fi

     for i in `seq 0 $(($np-1))`
     do
         mkdir -p cond$i/dcd
     done

     check_complete
     if [ $skip -eq 0 ]; then
        echo "trj unmixing using CHARMM"
        $charmm itt=$itt name=$comp ncycles=$ncycles nrep=$np < trj_unmixing.inp > /dev/null
        check_complete
     fi

     if [ $skip -eq 1 ]; then
        echo "trj unmixing is done!"
        rm -f aa*/dcd/${comp}_prod$itt.dcd
     fi

done

echo "done"
