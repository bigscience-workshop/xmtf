# Crosslingual Generalization through Multitask Finetuning

![](xmtf_banner.png)

This repository provides an overview of all components used for the creation of BLOOMZ & mT0 and xP3 introduced in the paper [Crosslingual Generalization through Multitask Finetuning](TODO).

### Data

- [xP3](https://huggingface.co/datasets/bigscience/xP3) created via TODO
- [xP3mt](https://huggingface.co/datasets/bigscience/xP3mt)

### Models

- BLOOMZ
    - [176B parameters](https://huggingface.co/bigscience/bloomz)
- mT0
    - ...

### Training

- [bigscience-workshop/Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed) using the scripts in [bigscience-workshop/bigscience](https://github.com/bigscience-workshop/bigscience/tree/master/train) filed under `tr13`

### Evaluation

- Logprobability tasks: []
- Generation Tasks: [bigscience-workshop/lm-evaluation-harness](https://github.com/bigscience-workshop/lm-evaluation-harness)
- Results are all available in this repository: https://huggingface.co/datasets/bigscience/evaluation-results

### Plots & Tables

- Scripts for all automatically created plots & tables in the paper are in `plotstables/`

