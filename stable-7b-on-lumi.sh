#!/bin/bash
#SBATCH --job-name=stable-7b
#SBATCH --account=project_462000229
#SBATCH --output=/pfs/lustref1/flash/project_462000229/logs/%j.log
#SBATCH --nodes=2               # Total number of nodes
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=6
#SBATCH --time=00:15:00
#SBATCH --mem=0			# All memory on the node
#SBATCH --partition=small-g

module purge
module load CrayEnv
module load PrgEnv-cray/8.3.3
module load craype-accel-amd-gfx90a
module load rocm/5.2.3
module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
module load aws-ofi-rccl/rocm-5.2.3

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1

# We need to set this to avoid "Cassini Event Queue overflow detected." errors.
export FI_CXI_DEFAULT_CQ_SIZE=131072

#export NCCL_DEBUG=INFO
export PYTHONPATH=.:${PYTHONPATH}
export WANDB_PROJECT=stable-7b
export ROCM_PATH=/opt/rocm

# Try playing with max_split_size_mb if you run into OOM errors.
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

export INTERCONNECT_PATH=/pfs/lustrep2/projappl/project_462000125/samantao-public/apps-rocm-5.2.3/aws-ofi-rccl
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:$INTERCONNECT_PATH

srun \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --distribution=block:block \
  --kill-on-bad-exit \
  run_with_environment.sh \
    singularity exec \
    -B"$PROJECT_DIR:$PROJECT_DIR" \
    -B"$SCRATCH_DIR:$SCRATCH_DIR" \
    -B"$FLASH_DIR:$FLASH_DIR" \
    -B"$INTERCONNECT_PATH:$INTERCONNECT_PATH" \
    -B /opt/cray:/opt/cray \
    -B /usr/lib64/libcxi.so.1:/usr/lib64/libcxi.so.1 \
    -B /usr/lib64/libjson-c.so.3:/usr/lib64/libjson-c.so.3 \
    $PROJECT_DIR/containers/mosaic-lumi_latest.sif \
    composer examples/llm/main.py examples/llm/yamls/mosaic_gpt/stable-7b.yaml \
      --run_name=${SLURM_JOB_ID} \
      --save_folder=${SCRATCH_DIR}/checkpoints/stable-7b
