# Crosslingual Generalization through Multitask Finetuning

![](xmtf_banner.png)

This repository provides an overview of all components used for the creation of BLOOMZ & mT0 and xP3 introduced in the paper [Crosslingual Generalization through Multitask Finetuning](https://arxiv.org/abs/2211.01786).

<!-- TOC -->

- [Data](#data)
- [Models](#models)
- [Create xP3](#create-xp3)
- [Train models](#train-models)
    - [BLOOMZ](#bloomz)
    - [mT0](#mt0)
- [Evaluate models](#evaluate-models)
    - [Rank Evaluation](#rank-evaluation)
    - [Generation Evaluation](#generation-evaluation)
- [Plots & Tables](#plots--tables)
    - [Plots](#plots)
    - [Tables](#tables)
- [Citation](#citation)

<!-- /TOC -->

## Data

<table>
  <tr>
<th>Name</th>
<th>Explanation</th>
<th>Example models</th>
</tr>
<tr>
<td><a href=https://huggingface.co/datasets/Muennighoff/xP3x>xP3x</a></t> 
<td>Mixture of 17 tasks in 277 languages with English prompts</td>
<td>WIP - Join us at Project Aya @<a href=https://cohere.for.ai/>C4AI</a> to help!</td>
</tr>
<tr>
<td><a href=https://huggingface.co/datasets/bigscience/xP3>xP3</a></t> 
<td>Mixture of 13 training tasks in 46 languages with English prompts</td>
<td><a href=https://huggingface.co/bigscience/bloomz>BLOOMZ</a> & <a href=https://huggingface.co/bigscience/mt0-xxl>mT0-13B</a></td>
</tr>
<tr>
<td><a href=https://huggingface.co/datasets/bigscience/xP3mt>xP3mt</a></t> 
<td>Mixture of 13 training tasks in 46 languages with prompts in 20 languages (machine-translated from English)</td>
<td><a href=https://huggingface.co/bigscience/bloomz-mt>BLOOMZ-MT</a> & <a href=https://huggingface.co/bigscience/mt0-xxl-mt>mT0-13B-MT</a></td>
</tr>
<tr>
<td><a href=https://huggingface.co/datasets/bigscience/xP3all>xP3all</a></t> 
<td>xP3 + our evaluation datasets adding an additional 3 tasks for a total of 16 tasks in 46 languages with English prompts</td>
<td></td>
</tr>
<tr>
<td><a href=https://huggingface.co/datasets/bigscience/xP3megds>xP3megds</a></t> 
<td><a href=https://github.com/bigscience-workshop/Megatron-DeepSpeed>Megatron-DeepSpeed</a> processed version of xP3</td>
<td><a href=https://huggingface.co/bigscience/bloomz>BLOOMZ</a></td>
</tr>
<tr>
<td><a href=https://huggingface.co/datasets/Muennighoff/P3>P3</a></t> 
<td>Repreprocessed version of the English-only <a href=https://huggingface.co/datasets/bigscience/P3>P3</a> with 8 training tasks</td>
<td><a href=https://huggingface.co/bigscience/bloomz-p3>BLOOMZ-P3</a> & <a href=https://huggingface.co/bigscience/mt0-xxl-p3>mT0-13B-P3</a></td>
</tr>
</table>

## Models

<table>
  <tr>
<th colspan="12">Multitask finetuned on <a style="font-weight:bold" href=https://huggingface.co/datasets/bigscience/xP3>xP3</a>. Recommended for prompting in English.
</tr>
<tr>
<td>Parameters</td>
<td>300M</td>
<td>580M</td>
<td>1.2B</td>
<td>3.7B</td>
<td>13B</td>
<td>560M</td>
<td>1.1B</td>
<td>1.7B</td>
<td>3B</td>
<td>7.1B</td>
<td>176B</td>
</tr>
<tr>
<td>Finetuned Model</td>
<td><a href=https://huggingface.co/bigscience/mt0-small>mt0-small</a></td>
<td><a href=https://huggingface.co/bigscience/mt0-base>mt0-base</a></td>
<td><a href=https://huggingface.co/bigscience/mt0-large>mt0-large</a></td>
<td><a href=https://huggingface.co/bigscience/mt0-xl>mt0-xl</a></td>
<td><a href=https://huggingface.co/bigscience/mt0-xxl>mt0-xxl</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz-560m>bloomz-560m</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz-1b1>bloomz-1b1</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz-1b7>bloomz-1b7</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz-3b>bloomz-3b</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz-7b1>bloomz-7b1</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz>bloomz</a></td>
</tr>
</tr>
  <tr>
<th colspan="12">Multitask finetuned on <a style="font-weight:bold" href=https://huggingface.co/datasets/bigscience/xP3mt>xP3mt</a>. Recommended for prompting in non-English.</th>
</tr>
<tr>
<td>Finetuned Model</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td><a href=https://huggingface.co/bigscience/mt0-xxl-mt>mt0-xxl-mt</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td><a href=https://huggingface.co/bigscience/bloomz-7b1-mt>bloomz-7b1-mt</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz-mt>bloomz-mt</a></td>
</tr>
<th colspan="12">Multitask finetuned on <a style="font-weight:bold" href=https://huggingface.co/datasets/Muennighoff/P3>P3</a>. Released for research purposes only. Strictly inferior to above models!</th>
</tr>
<tr>
<td>Finetuned Model</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td><a href=https://huggingface.co/bigscience/mt0-xxl-p3>mt0-xxl-p3</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td><a href=https://huggingface.co/bigscience/bloomz-7b1-p3>bloomz-7b1-p3</a></td>
<td><a href=https://huggingface.co/bigscience/bloomz-p3>bloomz-p3</a></td>
</tr>
<th colspan="12">Original pretrained checkpoints. Not recommended.</th>
<tr>
<td>Pretrained Model</td>
<td><a href=https://huggingface.co/google/mt5-small>mt5-small</a></td>
<td><a href=https://huggingface.co/google/mt5-base>mt5-base</a></td>
<td><a href=https://huggingface.co/google/mt5-large>mt5-large</a></td>
<td><a href=https://huggingface.co/google/mt5-xl>mt5-xl</a></td>
<td><a href=https://huggingface.co/google/mt5-xxl>mt5-xxl</a></td>
<td><a href=https://huggingface.co/bigscience/bloom-560m>bloom-560m</a></td>
<td><a href=https://huggingface.co/bigscience/bloom-1b1>bloom-1b1</a></td>
<td><a href=https://huggingface.co/bigscience/bloom-1b7>bloom-1b7</a></td>
<td><a href=https://huggingface.co/bigscience/bloom-3b>bloom-3b</a></td>
<td><a href=https://huggingface.co/bigscience/bloom-7b1>bloom-7b1</a></td>
<td><a href=https://huggingface.co/bigscience/bloom>bloom</a></td>
</tr>
</table>

## Create xP3(x)

We have processed & uploaded [xP3](https://huggingface.co/datasets/bigscience/xP3). If you want to recreate it, follow these steps:

1. Get promptsource: For xP3mt `git clone -b xp3mt https://github.com/Muennighoff/promptsource.git`, for xP3 `git clone -b tr13 https://github.com/Muennighoff/promptsource.git` & install `cd promptsource; pip install -e .`
2. Get packages `pip install -q datasets iso-639`
3. Get the [creation script](https://github.com/bigscience-workshop/bigscience/blob/master/data/xp3/prepare_xp3_train.py) & edit it if necessary:
- For xP3mt, set `USE_ENGLISH_PROMPTS = False` in the beginning
- For xP3, set `USE_ENGLISH_PROMPTS = True` in the beginning
4. Run the script, such as via `python prepare_xp3.py` or a [SLURM script](https://github.com/bigscience-workshop/bigscience/blob/master/data/xp3/prepare_xp3_train.slurm)

For the new extension of xP3, [xP3x](https://huggingface.co/datasets/Muennighoff/xP3x), the process is largely the same except:

1. Install the `xp3` branch instead i.e. `pip install git+https://github.com/Muennighoff/promptsource.git@xp3x`
3. The creation script is in this repository & named `create_xp3x.py`.

xP3x is a superset of xP3, so unless you want to reproduce the paper, we recommend always using xP3x (or xP3mt if you want machine-translated prompts).

## Train models

### BLOOMZ

1. Download the pretrained model [checkpoint](https://huggingface.co/bigscience/bloom-optimizer-states), which is of shape PP=12, TP=4, DP=4. If you'd like to reshape the model you will also need to download [the universal checkpoint](https://huggingface.co/bigscience/bloom-optimizer-states/tree/global_step95000_universal). If you want to continue finetuning, you should use [our finetuned checkpoint](https://huggingface.co/bigscience/bloomz-optimizer-states), which is of shape PP=72, TP=1, DP=4.
2. Setup the training code: `git clone -b t0loading https://github.com/bigscience-workshop/Megatron-DeepSpeed` & follow its [setup guide](https://github.com/bigscience-workshop/Megatron-DeepSpeed/tree/t0loading#get-started-fast) to create an environment with necessary packages.
3. Download the Megatron-DeepSpeed processed [xP3megds](https://huggingface.co/datasets/bigscience/xP3megds) or repreprocess it for Megatron-DeepSpeed yourself by downloading [xP3](https://huggingface.co/datasets/bigscience/xP3), removing the `merged_{lang}.jsonl` files & preprocess it using the script [here](https://github.com/bigscience-workshop/bigscience/blob/master/data/xp3/xp3_jsonl_to_meg.slurm).
4. Setup & run the training script: We use SLURM scripts available at [bigscience-workshop/bigscience/train/tr13-mtf](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr13-mtf) and referred to as `xp3capmixnewcodelonglossseq`. E.g. [this is the script launched to train bloomz](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr13-mtf/tr13-176B-mtf-xp3capmixnewcodelonglossseq.slurm). Important parts of the script to modify are:
- `#SBATCH` variables, such as nodes, gpus, time, etc. - Our SLURM guide is [here](https://github.com/bigscience-workshop/bigscience/tree/master/jz/slurm#slurm-how-to)
- `source $six_ALL_CCFRWORK/start-tr13f-6B3-ml-t0` to point to your own conda environment setup via Megatron-DeepSpeed
- PATH environment variables, notably
    - `TRAIN_DATA_PATH` & `VALID_DATA_PATH`, which point to files pointing to your processed training and validation data. We provide our files in this repository (`xp3capmixnewcodelong_train.txt` & `xp3capmixnewcodelong_validation.txt`), but you will likely want to change the paths inside. The percentages per language are based on how much each language makes up in xP3 with code being slightly upsampled.
- PP_SIZE=72, TP_SIZE=1 & BATCH SIZE & co specifying the layout. This will depend on the hardware available to you. If you change, you may have to reshape the model. For reshaping you need to use the universal checkpoint and use the `--universal` flag in the script. We recommend saving a new checkpoint right after & then continuing training without `--universal`, which will be faster.
- If you want to restart from a saved checkpoint (e.g. after training a few steps like above), make sure to remove the `--no-load-optim` & `--reset-progress` flags
- After training, you can convert the checkpoint to transformers format using the script [here](https://github.com/huggingface/transformers/blob/ee8e80a060d65ab349743ffcb5842365eb0e5606/src/transformers/models/bloom/convert_bloom_original_checkpoint_to_pytorch.py)

Helpful resources:
- [Blog post](https://huggingface.co/blog/bloom-megatron-deepspeed)
- BLOOM community tab, such as [here](https://huggingface.co/bigscience/bloom/discussions/46)

### mT0

Follow the finetuning instructions [here](https://github.com/google-research/t5x/blob/main/docs/usage/finetune.md) making sure to use pretrained mT5 models & the xP3 dataset.

Helpful resources:
- [T5X paper](https://arxiv.org/abs/2203.17189)

## Evaluate models

Evaluation results are all available in this repository: https://huggingface.co/datasets/bigscience/evaluation-results under the respective models.
Below we explain how to run evaluation.

### Rank Evaluation 

We evaluate the models on Rank Evaluation on [XCOPA](https://huggingface.co/datasets/xcopa), [XNLI](https://huggingface.co/datasets/xnli), [XStoryCloze](https://huggingface.co/datasets/Muennighoff/xstory_cloze) & [XWinograd](https://huggingface.co/datasets/Muennighoff/xwinograd):

1. Get promptsource fork: `git clone -b xp3mt https://github.com/Muennighoff/promptsource.git` & `cd promptsource; pip install -e .`
2. Get t-zero fork: `git clone -b muennighoff/upgrdps https://github.com/Muennighoff/t-zero.git` & `cd t-zero; pip install -e .`
3. Download model & run evaluation script, for example for [bloomz](https://github.com/bigscience-workshop/bigscience/blob/master/evaluation/results/tr13/tzeroeval/evaluate_t0_176b.slurm).

### Generation Evaluation

We evaluate generation on translation & summarization during training for validation:

1. Get promptsource fork: `git clone -b xp3mt https://github.com/Muennighoff/promptsource` & `cd promptsource; pip install -e .`
2. Get [bigscience-workshop/lm-evaluation-harness](https://github.com/bigscience-workshop/lm-evaluation-harness): `git clone https://github.com/bigscience-workshop/lm-evaluation-harness`. The script for the 7.1B model, for example, is [here](https://github.com/bigscience-workshop/bigscience/blob/master/evaluation/results/tr13/lmeval/run_generation_7b1.slurm).

We also evaluate code generation on [HumanEval](https://huggingface.co/datasets/openai_humaneval):

1. Get code evaluation code `git clone https://github.com/loubnabnl/bloom-code-evaluation` & go through its setup.
2. Set `prepend_eos` to `False` in `code_eval.py` at `complete_code(model, tokenizer, prompt, num_completions=1, prepend_eos=True, **gen_kwargs)` i.e. `complete_code(model, tokenizer, prompt, num_completions=1, prepend_eos=False, **gen_kwargs)`.
3. Download model & run evaluation script swapping out MODEL_CKPT for your path, for example for bloomz use [this](https://github.com/loubnabnl/bloom-code-evaluation/blob/master/generate_code_bloom.slurm).


## Plots & Tables

### Plots

- Figure 1: `plotstables/xp3_taxonomy.drawio` & `plotstables/xp3_taxonomy.pdf`
- Figure 2: `plotstables/xp3_languages.ipynb` & [colab](https://colab.research.google.com/drive/1yRDXktu030DnipFBj6-dwOGNVIdgktA9?usp=sharing)
- Figure 3: `plotstables/xp3_variants.pdf` & [drawings](https://docs.google.com/drawings/d/1wSt_X0olUFcOFQ5D1UnMv1V-LKMr3WZIRIgaFypTP24/edit?usp=sharing)
- Figure 4: `plotstables/xp3_generalization_bar.pdf` & [colab](https://colab.research.google.com/drive/1bz083LuBJi0-pLOqdr4_ycEctn6obYST?usp=sharing)
- Figure 5: `plotstables/lang_generalization` & [colab](https://colab.research.google.com/drive/1lFFR6_ijR_iWJQnqIW5y5-LuRnRoRTS3?usp=sharing)
- Figure 6: `plotstables/scale.pdf` & [colab](https://colab.research.google.com/drive/19GcYT5SJFpyu8B0RrewN462w3i461mZ5?usp=sharing)
- Figure 7: `plotstables/validation.pdf` & [colab](https://colab.research.google.com/drive/1FWW7LMKC9kQNLgCLZXl_dBER5wBSPGMu?usp=sharing)
- Figure 8: `plotstables/pretraining_sizes.pdf` & [colab](https://colab.research.google.com/drive/1hpW6xEnU56Ed7DmXrREzczGwEeNV8KJ2?usp=sharing)
- Figure 9: `plotstables/english_task_generalization.pdf` & [colab](https://colab.research.google.com/drive/1lFFR6_ijR_iWJQnqIW5y5-LuRnRoRTS3?usp=sharing)
- Figure 10: `plotstables/task_generalization.pdf` & [colab](https://colab.research.google.com/drive/1lFFR6_ijR_iWJQnqIW5y5-LuRnRoRTS3?usp=sharing)
- Figure 11: `plotstables/roots_xp3_languages.pdf` & [colab](https://colab.research.google.com/drive/1ankXUcTqjPantCzIfUSwAjYfAhkR7M6o?usp=sharing) requiring some of the files in `plotstables/contamination`
- Figure 12: `plotstables/examples/bloom_code_example.py` & `plotstables/examples/bloom_code_light.pdf` & `plotstables/examples/bloomz_code_light.pdf`; The raw code files can be found [here](https://huggingface.co/datasets/bigscience/evaluation-results/blob/main/bloom/codeeval/transformers/openai_humaneval/code_generations_bloom.zip) & [here](https://huggingface.co/datasets/bigscience/evaluation-results/blob/main/bloomz/codeeval/transformers/openai_humaneval/code_generations_bloomz.zip)
- Figure 13 - Figure 16: `plotstables/examples/*.pdf` & `plotstables/examples/generations.drawio`

### Tables

- Table 1: [Colab](https://colab.research.google.com/drive/1ZhwHDaHBPUlZiTp-ZZxy7axuWgE68FkW?usp=sharing) & [Colab for complex version](https://colab.research.google.com/drive/1WCUgfjToVJ9b_fJHzkWKsuGzVofqv38x?usp=sharing)
- Table 2: Adapted from the Codex paper
- Table 3: Manual
- Table 4: `plotstables/compute_codegen_len.ipynb` for generations & `plotstables/countcode.py` for xP3
- Table 5: Manual
- Table 6: Manual
- Table 7: `plotstables/levenshtein.py`
- Table 8: Same as Table 1 with languages swapped from L1 to L2
- Table 9: [Colab](https://colab.research.google.com/drive/1AWJk3jbrD1VpiMARW-xATalrupwFzZN-?usp=sharing)
- Table 10: [Colab](https://colab.research.google.com/drive/14t9w6QSf2K5BQP0cInyGsreAhY271DLB?usp=sharing)
- Prompt Appendix: https://github.com/albanie/prompt_formatting_in_latex

## Citation

```bibtex
@article{muennighoff2022crosslingual,
  title={Crosslingual generalization through multitask finetuning},
  author={Muennighoff, Niklas and Wang, Thomas and Sutawika, Lintang and Roberts, Adam and Biderman, Stella and Scao, Teven Le and Bari, M Saiful and Shen, Sheng and Yong, Zheng-Xin and Schoelkopf, Hailey and others},
  journal={arXiv preprint arXiv:2211.01786},
  year={2022}
}
```
