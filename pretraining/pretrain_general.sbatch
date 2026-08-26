#!/bin/bash
#SBATCH --job-name=OnlyNByN
#SBATCH --cpus-per-task=1
#SBATCH --mincpus=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:4g.47gb:1
#SBATCH --time=18:00:00
#SBATCH --output=onlynbynNEW_%j.out
#SBATCH --error=onlynbynNEW_%j.err

# ------------------------------
# Load HF token
# ------------------------------
export HF_TOKEN=$(cat /scratch/wss37/.hf_token)
export HF_HOME=/scratch/wss37/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"             # huggingface_hub repo cache
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

echo "Using Transformers Cache: ${TRANSFORMERS_CACHE}"

echo "Using HF token: ${HF_TOKEN:0:6}*****"   # prints only part of it

# ------------------------------
# Load environment
# ------------------------------
# module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/wss37/.venvs/gpt-bert

# ------------------------------
# Navigate & run your script
# ------------------------------
cd /scratch/wss37/gpt-bert/gpt-bert/pretraining/

echo "HOSTNAME=$HOSTNAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
nvidia-smi
python - <<'PY'
import torch, os
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

# Run your prediction script
python3 train_single_gpu.py --train_path "../data/train_onlynbyn_short_onlynbyn_tokenized.bin" \
    --valid_path "../data/dev_short_onlynbyn_tokenized.bin" \
    --name "onlynbyn_100M_short_3" \
    --tokenizer_path "../tokenizers/onlynbyn_short.json" \
    --seed 3 \
    --global_batch_size 256 \
    --local_batch_size 256 \
    --max_steps 60_000 \
    --learning_rate 3e-4 \
    --hybrid_numerator 10 \
    --hybrid_denominator 20

