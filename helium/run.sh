#!/bin/bash
#SBATCH --job-name=agn_rates
#SBATCH --output=logs/rates_%A_%a.out  # %A é o ID principal do array, %a é o sub-ID (0, 1, 2)
#SBATCH --error=logs/rates_%A_%a.err
#SBATCH --array=0-2                    # Cria 3 jobs simultâneos (índices 0, 1 e 2)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=120:00:00                # Ajustado para 72h devido à carga
#SBATCH --partition=cpu_long

# Ativação do ambiente
source $(conda info --base)/etc/profile.d/conda.sh
conda activate FR0cat

# Exporta variáveis para o joblib não extrapolar os núcleos
export LOKY_MAX_CPU_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Iniciando Job Array ID: $SLURM_ARRAY_TASK_ID em $(date)"
python helium.py
echo "Finalizado em $(date)"