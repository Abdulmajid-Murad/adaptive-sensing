#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --job-name="adpative-sensing"
#SBATCH --mail-user=abdulmajid.a.murad@ntnu.no
#SBATCH --mail-type=END
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --mem-per-cpu=16gb         # Memory per processor
#SBATCH --time=24:00:00             # Time limit hrs:min:sec
#SBATCH --output=array_%A-%a.out    # Standard output and error log
#SBATCH --array=0-64                # Array range




WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"


module purge
module load GCC/8.3.0
module load CUDA/10.1.243
module load OpenMPI/3.1.4
source ~/prob/bin/activate
module load TensorFlow/1.15.0-Python-3.7.4
echo "after modules are loaded"

python run_experiment.py 
echo "Done."
deactivate
