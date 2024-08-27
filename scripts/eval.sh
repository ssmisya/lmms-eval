source ~/.bashrc
source ~/anaconda3/bin/activate share4v

# environment variables
export OMP_NUM_THREADS=8
AD_NAME=songmingyang
encrypted_password=iWRsYqbwV4EJgJvU8QjLe00CptZc5jBVH3FMo5i6n9mVdOSoUurpyBTmst1Z
new_proxy_address=http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address
code_base=/mnt/petrelfs/songmingyang/code/tools/lmms-eval/scripts
cd $code_base
export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT
export SLURM_JOB_ID=3458700 
# unset SLURM_JOB_ID       
config_file=$1

export API_TYPE=openai
export OPENAI_API_URL=https://kapkey.chatgptapi.org.cn/v1/chat/completions
export OPENAI_API_KEY=sk-N2kUpO5iy8Hh60PuBd09697fEaA549488eC2B961Af48E2Bd

gpus=8
cpus=64
quotatype="reserved"
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="eval" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
accelerate launch  --config_file  ./accelerate_config.yaml \
-m lmms_eval --config  ${config_file} \
--verbosity INFO


# salloc --partition=MoE --job-name="interact" --gres=gpu:8 -n1 --ntasks-per-node=1 -c 64 --quotatype="reserved"
# salloc --partition=MoE --job-name="interact" --gres=gpu:1 -n1 --ntasks-per-node=1 -c 16 --quotatype="reserved"
