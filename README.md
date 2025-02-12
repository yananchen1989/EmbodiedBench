<h1 align="center">
<img src="docs/images/embodied-logo.png" alt="embodied-logo" width="40" height="40" style="vertical-align: middle; margin-top: -12px;">
EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents
</h1>


<p align="center">
  📄  <a href="docs/pdf/paper.pdf"><strong>Paper</strong></a> |  
  🏠 <a href="https://embodiedbench.github.io"><strong>Project Website</strong></a>
</p>


<p align="center">
    <a href="https://yangrui2015.github.io/">Rui Yang*</a>, 
    <a href="">Hanyang Chen*</a>, 
    <a href="https://jyzhang1208.github.io/">Junyu Zhang*</a>, 
    <a href="">Mark Zhao*</a>, 
    <a href="https://qiancheng0.github.io/">Cheng Qian</a>, 
    <a href="https://jameskrw.github.io/">Kangrui Wang</a>, 
    <a href="https://qinengwang-aiden.github.io/">Qineng Wang</a>, 
    <a href="">Teja Venkat Koripella</a>, 
    <a href="">Marziyeh Movahedi</a>, 
    <a href="https://limanling.github.io/">Manling Li</a>, 
    <a href="https://blender.cs.illinois.edu/hengji.html">Heng Ji</a>, 
    <a href="https://www.huan-zhang.com/">Huan Zhang</a>, 
    <a href="https://tongzhang-ml.org/">Tong Zhang</a>
</p>
<p align="center">University of Illinois Urbana-Champaign, Northwestern University, University of Toronto, Toyota Technological Institute at Chicago</p>


<img src="docs/images/framework.png" width="100%" />

# 🔥 Overview 
We introduce **EmbodiedBench**, a comprehensive benchmark designed to evaluate **Multi-modal Large Language Models (MLLMs) as embodied agents**. While existing benchmarks have primarily focused on Large Language Models (LLMs) and high-level tasks, EmbodiedBench takes a leap forward by offering a comprehensive, fine-grained evaluation of MLLM-based agents across both **high-level and low-level tasks**, as well as **six critical agent capabilities**. 

EmbodiedBench is more than just a benchmark—it’s a **multifaceted, standardized evaluation platform** that not only uncovers the current challenges in embodied AI but also provides actionable insights to push the boundaries of MLLM-driven embodied intelligence.  


## 🚀 **Key Features** 

- 🛠️ **Diverse Tasks with Hierarchical Action Levels:**  
  **1,128 testing tasks** across four environments, spanning from high-level tasks (EB-ALFRED and EB-Habitat) to low-level tasks (EB-Navigation and EB-Manipulation). We created new high-quality datasets and enhanced existing simulators to support comprehensive assessments.

- 🎯 **Capability-Oriented Evaluation:**  
  **Six specialized subsets** to evaluate essential agent capabilities, including commonsense reasoning, complex instruction, spatial awareness, visual perception, and long-term planning. 

- ⚡ **Unified APIs for Embodied Environments:**  
  EmbodiedBench provides **[Gym](https://github.com/openai/gym)-style APIs for all environments**, ensuring ease of use and seamless agent evaluation.  

- 🏹 **Effortless MLLM/LLM Evaluation (API & Local Support):**  
  - Supports **proprietary** (e.g., OpenAI API) and **open-source models** (local execution).  
  - Enables self-hosted model evaluation using OpenAI API-style calls or offline execution based on [LMDeploy](https://github.com/InternLM/lmdeploy).  
  - While mainly focused on MLLMs, EmbodiedBench also supports **LLM evaluation**.  


- 🔧 **Configurable Textual and Visual Designs:**  
Our flexible configuration options enable in-depth experimentation with **visual input**, **textual and visual in-context prompts**, **environment feedback**, **camera resolution**, **detection boxes**, and **multi-step/multi-view image inputs** and more, empowering researchers to better understand the role of each component in agent performance.


### Planning Examples in EB-ALFRED and EB-Manipulation
<img src="docs/images/planning_example.png" width="100%" />


### Comparison with related benchmarks

"Fine-grained" indicates a multi-dimensional evaluation approach rather than an overall accuracy.  
¹AgentBench and VisualAgentBench include domains such as household, games, and web.  
²VLABench is originally used for evaluating VLA models.

| Benchmark | Category | Action Level | #Env. | #Test Tasks | Multimodal | Fine-grained | LLM/VLM Support |
|-----------|----------|--------------|------|-------------|------------|--------------|-----------------|
| Alfred  | Household | High | 1 | 3062 | ✅ | ❌ | ❌ |
| VLMbench  | Manipulation | Low | 1 | 4760 | ✅ | ❌ | ❌ |
| Language Rearrangement  | Household | High | 1 | 1000 | ✅ | ✅ | ❌ |
| GOAT-bench  | Navigation | Low | 1 | 3919 | ✅ | ❌ | ❌ |
| AgentBench  | Multi-domain¹ | High | 8 | 1091 | ❌ | ❌ | ✅ |
| Lota-bench  | Household | High | 2 | 308 | ❌ | ❌ | ✅ |
| VisualAgentBench  | Multi-domain¹ | High | 5 | 746 | ✅ | ❌ | ✅ |
| Embodied Agent Interface  | Household | High | 2 | 438 | ❌ | ✅ | ✅ |
| VLABench  | Manipulation | Low² | 1 | 100 | ✅ | ✅ | ✅ |
| **EmbodiedBench (ours)** | Multi-domain | High & Low | 4 | 1128 | ✅ | ✅ | ✅ |



# 🖥️ Installation
**Note: we will install three conda environments one for EB-ALFRED and EB-Habitat, one for EB-Navigation, and one for EB-Manipulation.**

Download repo
```
git clone https://github.com/EmbodiedBench/EmbodiedBench.git
cd EmbodiedBench
```

1️⃣ Environment for ```Habitat and Alfred```
```
conda env create -f conda_envs/environment.yaml 
conda activate embench
pip install -e .
```
2️⃣ Environment for ```EB-Navigation```
```
conda env create -f conda_envs/environment_eb-nav.yaml 
conda activate embench_nav
pip install -e .
```
3️⃣ Environment for ```EB-Manipulation```
```
conda env create -f conda_envs/environment_eb-man.yaml 
conda activate embench_man
pip install -e .
```

## Start Headless Server
Please run startx.py script before running experiment on headless servers. We use X_DISPLAY id=1 by default.
```
python -m embodiedbench.envs.eb_alfred.scripts.startx 1
```

## EB-Alfred

Run the following code to ensure the EB-ALFRED environment is working correctly.
```
conda activate embench
python -m embodiedbench.envs.eb_alfred.EBAlfEnv
```

## EB-Habitat

- Install [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) via

 ```
 conda activate embench
 conda install -y habitat-sim==0.3.0 withbullet  headless -c conda-forge -c aihabitat
 git clone -b 'v0.3.0' --depth 1 https://github.com/facebookresearch/habitat-lab.git ~/habitat-lab
 cd ~/habitat-lab
 pip install -e habitat-lab
 pip install -e habitat-baselines
 ## now the above habitat-baselines can lead to wrong version of numpy, just reinstall it
 pip install numpy==1.23.5 
 ```

- Download YCB and ReplicaCAD dataset for the Language Rearrangement task. 
> **Remember to run the following commands inside the eb_habitat folder; otherwise, errors would occur.**
```
cd embodiedbench/embodiedbench/envs/eb_habitat
conda install -y -c conda-forge git-lfs
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets
cd ../..
```

Run the following code to ensure the EB-Habitat environment is working correctly.
```
python -m embodiedbench.envs.eb_habitat.EBHabEnv
```

## EB-Navigation

Run the following code to ensure the EB-Navigation environment is working correctly.
```
conda activate embench_nav
python -m embodiedbench.envs.eb_navigation.EBNavEnv
```

## EB-Manipulation
* Install Coppelia Simulator

CoppeliaSim V4.1.0 required for Ubuntu 20.04; you can find other versions here (https://www.coppeliarobotics.com/previousVersions#)

```bash
conda activate embench_man
cd embodiedbench/envs/eb_manipulation
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
rm CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
mv CoppeliaSim_Pro_V4_1_0_Ubuntu20_04/ /PATH/YOU/WANT/TO/PLACE/COPPELIASIM
```

* Add the following to your *~/.bashrc* file:

```bash
export COPPELIASIM_ROOT=/PATH/YOU/WANT/TO/PLACE/COPPELIASIM
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

> Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

* Install the PyRep, EB-Manipulation package and dataset:
```bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install -e .
cd ..
pip install -r requirements.txt
pip install -e .
cp ./simAddOnScript_PyRep.lua $COPPELIASIM_ROOT
git clone https://huggingface.co/datasets/markyyds/eb_manipulation_dataset
mv eb_manipulation_dataset/data/ ./
rm -rf eb_manipulation_dataset/
```

> Remember that whenever you re-install the PyRep, simAddOnScript_PyRep.lua will be overwritten. Then, you should copy this again.

* Run the following code to ensure the EB-Manipulation is working correctly (set display to :0 if you have not):
```bash
cd ../../..
export DISPLAY=:0
python -m embodiedbench.envs.eb_manipulation.EBManEnv
```


# 🚀 Quick Start
### Proprietary Models
Before running evaluations, set up your environment variables if you plan to use proprietary models:
```
export OPENAI_API_KEY="your_oai_api_key_here"
export GEMINI_API_KEY="your_gemini_api_key_here"
export ANTHROPIC_API_KEY="your_anpic_api_key_here"
```
To evaluate MLLMs in EmbodiedBench, activate the corresponding Conda environment and run:
```
conda activate embench
python -m embodiedbench.main env=eb-alf model_name=gpt-4o-mini exp_name='baseline'
python -m embodiedbench.main env=eb-hab model_name=gpt-4o-mini exp_name='baseline'

conda activate embench_nav
python -m embodiedbench.main env=eb-nav model_name=gpt-4o exp_name='baseline'

conda activate embench_man 
python -m embodiedbench.main env=eb-man model_name=claude-3-5-sonnet-20241022 exp_name='baseline'
```
#### Configuration Options
You can customize the evaluation using the following flags:
- **`env`**: The environment to test. Choose from:  
  - `'eb-alf'` (EB-ALFREd)  
  - `'eb-hab'` (EB-Habitat)  
  - `'eb-man'` (EB-Manipulation)  
  - `'eb-nav'` (EB-Navigation)  

- **`model_name`**: Full model name, including proprietary options like:  
  - `'gpt-4o'`, `'gpt-4o-mini'`, `'claude-3-5-sonnet-20241022'`, `'gemini-1.5-pro'`, `'gemini-2.0-flash-exp'`, `'gemini-1.5-flash'`  

- **`model_type`**: Set to `'remote'` by default.  
- **`down_sample_ratio`**: Data sampling ratio (default `1.0`). Use `0.1` for debugging (10% of the dataset).  
- **`language_only`**: If `True` (or `1`), the agent receives only text input (default: `False`).  
- **`eval_sets`**: List of subsets to evaluate (default: all subsets).  
- **`chat_history`**: Enables multi-turn interaction (`False` by default, as it may reduce performance).  
- **`n_shots`**: Maximum number of textual examples for in-context learning (varies by environment).  
- **`multiview`**: Uses multi-view images as input (only for EB-Manipulation & EB-Navigation, default: `False`).  
- **`multistep`**: Includes historical multi-step images (`False` by default).  
- **`detection_box`**: Enables detection box input (valid for EB-ALFREd, EB-Navigation, and EB-Manipulation).  
- **`resolution`**: Image resolution (default: `500`).  
- **`exp_name`**: Name of the experiment, used in logging.  
- **`visual_icl`**: Enables visual in-context learning (`False` by default).  
- **`log_level`**: Sets the logging level (`INFO` by default). Use `DEBUG` for debugging purpose.

> ⚠️ **Important:** Avoid enabling multiple flags simultaneously from `visual_icl`, `multiview`, `multistep`, and `chat_history` to prevent excessive image inputs and conflicts.  

---

### Open-source Models
We support two deployment methods for open-source models: **offline running** and **model serving**.  

#### **1️⃣ Offline Running**  
For local execution, set `model_type=local` and adjust `tp` (tensor parallelism) based on GPU memory.  
- A rough guideline: For 48GB GPUs, use `tp = ceil(model size (in B) / 10)`.  
```
conda activate embench
python -m embodiedbench.main env=eb-alf model_name=Qwen/Qwen2-VL-7B-Instruct model_type=local exp_name='baseline' tp=1
python -m embodiedbench.main env=eb-hab model_name=OpenGVLab/InternVL2_5-8B model_type=local exp_name='baseline' tp=1


conda activate embench_nav
python -m embodiedbench.main env=eb-nav model_name=OpenGVLab/InternVL2_5-38B model_type=local exp_name='baseline' tp=4

conda activate embench_man 
python -m embodiedbench.main env=eb-man model_name=meta-llama/Llama-3.2-11B-Vision-Instruct model_type=local exp_name='baseline' tp=2
```

#### **2️⃣ Online Model Serving (Recommended)**  
Model serving decouples **model execution** from **evaluation**, allowing flexible deployment via API calls.  
```
## Step 0, create an environment for lmdeploy
conda env create -f conda_envs/lmdeploy.yaml
conda activate lmdeploy
pip install lmdeploy

## Step 1, open another tmux window, runing the model
lmdeploy serve api_server "OpenGVLab/InternVL2_5-8B" --server-port $port --tp 1

## Step 2, running the evaluation
conda activate embench
export remote_url='$IP_address:$port/v1' # set the address for access
python -m embodiedbench.main env=eb-hab model_name=OpenGVLab/InternVL2_5-8B exp_name='baseline' 
```

## Docker
To be added.



# Acknowledgement
This repo is based on awesome embodied benchmarks and simulations [Lota-Bench](https://github.com/lbaa2022/LLMTaskPlanning), [ALFRED](https://github.com/askforalfred/alfred), [ai2thor](https://github.com/allenai/ai2thor), [EmbodiedAgentInterface](https://github.com/embodied-agent-interface/embodied-agent-interface), [ML-Llarp](https://github.com/apple/ml-llarp), [Habitat](https://github.com/facebookresearch/habitat-lab), [VLMBench](https://github.com/eric-ai-lab/VLMbench), and [RLBench](https://github.com/stepjam/RLBench). Our open-source model deployment is based on [LMDeploy](https://github.com/InternLM/lmdeploy). 

# Citation
```
@misc{yang2025embodiedbench,
      title={EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents}, 
      author={Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, Heng Ji, Huan Zhang, Tong Zhang},
      year={2025},
      eprint={2502},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```
