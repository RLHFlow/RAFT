# RAFT
This is an official implementation of the [Reward rAnked Fine-Tuning Algorithm (RAFT)](https://arxiv.org/pdf/2304.06767), also known as iterative best-of-n fine-tuning or rejection sampling fine-tuning.


## 1 Structure 

The initial release of tis project focus on the Bradley-Terry reward modeling and pairwise preference model. Since then, we have included more advanced techniques to construct preference model. The structure of this project is 

- [`Data generation`](./generation/) to generate n responses per prompt;
- [`Reward Ranking`](./annotate_data/) to compute the rewards of the responses and select the response with highest reward;
- [`Finetuning`](./sft/) to finetune the model on the selected responses.


We also provide a small demo for RAFT+diffusion model in [diffusion-example](./diffusion-example).

You may also refer to our [colab](https://colab.research.google.com/drive/1bQmlSiKnqFjrkijFUJ5ylbYW-zUwObqL) for more information.

## 2 Installation instructions

It is recommended to have two separate environments for inference and training, respectively.

**Inference Environment**

```sh
conda create -n vllm python=3.10.9
conda activate vllm
pip install datasets

# The following code is tested for CUDA12.0-12.2, and CUDA12.6
# To develop llama-3, mistral, gemma-1, 1.1, 2, deepseek you can consider the following vllm version
pip install vllm==0.5.4

pip install accelerate==0.33.0
pip install deepspeed==0.14.5
pip install transformers==4.43.4
pip install numpy==1.26.4 #Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!
```

**Training Environment**


```shell
conda create -n sft_train python=3.10.9
conda activate sft_train

## Get axolotl for general model
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
git checkout 55cc214c767741e83ee7b346e5e13e6c03b7b9fa
pip install -e .

# The test cuda version is 12.1, 12.2. You may need to update the torch version based on your cuda version...
# you may encounter underfined symbol error related to cuda and flash-attn and 2.1.2 can solve it ...
pip3 install torch==2.1.2 torchvision torchaudio
pip install flash-attn==2.6.3


## Get FastChat
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .

git clone https://github.com/WeiXiongUST/RLHF-Reward-Modeling.git
pip install deepspeed
```

## Running the code

### 3.1 Data generation

We have prepared some prompt sets on huggingface.
- UltraFeedback RLHFlow/ultrafeedback_iter1, RLHFlow/ultrafeedback_iter2, RLHFlow/ultrafeedback_iter3
- RLHFlow/iterative-prompt-v1-iter1-20K, RLHFlow/iterative-prompt-v1-iter2-20K, RLHFlow/iterative-prompt-v1-iter3-20K ...

To accelerate data generation, we use the VLLM. We prepare two ways of using VLLM to inference for a more robust implementation, where you can try them out and choose the one that fits with your environment best. We use LLaMA3-8B as an example. 

You may create a test_gen.sh file, and copy the following contents into the file and run ``bash test_gen.sh''.

```sh
# First approach: initialize 4 VLLM processes and split the prompt set to the 4 agents
# The generated samples will be stored at output_dir + local_index + ".jsonl

my_world_size=8 # how many gpu you use
infer_model=RLHFlow/LLaMA3-SFT
prompt_dir=RLHFlow/test_generation_2k
mkdir data
output_dir=./data/gen_data

conda activate vllm
CUDA_VISIBLE_DEVICES=0 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 0 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=1 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 1 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=2 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 2 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=3 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 3 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=4 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 4 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=5 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 5 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=6 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 6 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=7 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 7 --my_world_size ${my_world_size}  &

# then, we merge the 8 datasets into one dataset.
wait
python ./generation/merge_data.py --base_path ${output_dir} --output_dir ./data/gen_data.json --num_datasets ${my_world_size}
```

We can also use API server to generate new responses.

```sh
mkdir data
conda activate vllm

# register the api server
bash ./generation/register_server.sh RLHFlow/LLaMA3-SFT

# start to generate
python ./generation/gen_hf.py --ports 8000 8001 8002 8003 8004 8005 8006 8007 --tokenizer RLHFlow/LLaMA3-SFT --dataset_name_or_path RLHFlow/test_generation_2k --output_dir ./data/gen_data.jsonl --K 4 --temperature 1.0
```

### 3.2 Data Annotation
Then, we call the reward/preference model trained in step 2 to rank the generated responses. 

```sh
accelerate launch ./annotate_data/get_rewards.py --dataset_name_or_path ./data/gen_data.jsonl --output_dir ./data/data_with_rewards.jsonl --K 4

python ./annotate_data/get_bon_data.py --dataset_name_or_path ./data/data_with_rewards.jsonl --output_dir your_huggingface_dataset_dir
```

If you encounter error ``TypeError: Got unsupported ScalarType BFloat16'', considering adjusting your transformer version.

### 3.3 Training

```sh
conda activate sft_train
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node 8 --master_port 20001 -m axolotl.cli.train gemma-2b-it.yaml --deepspeed ./configs/deepspeed_stage2.json
```
If you encounter out-of-memory issue. Running the code with Gemma-2b-it with deepspeed stage 3 and gradient checkpoint (set in the config).

## Citation

If you find the content of this repo useful in your work, please consider citing:

```bibtex
@article{dong2023raft,
  title={{RAFT}: Reward rAnked FineTuning for Generative Foundation Model Alignment},
  author={Hanze Dong and Wei Xiong and Deepanshu Goyal and Yihan Zhang and Winnie Chow and Rui Pan and Shizhe Diao and Jipeng Zhang and KaShun SHUM and Tong Zhang},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023},
  url={https://openreview.net/forum?id=m7p5O7zblY},
}

```
