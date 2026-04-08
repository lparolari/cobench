# Benchmarking Layout-Guided Diffusion Models through Unified Semantic-Spatial Evaluation in Closed and Open Settings

This repository provides the benchmarks, evaluation tools, and model zoo used in our study on the evaluation of layout-guided text-to-image generative models. It includes both **closed-set** and **open-set** benchmarks, the evaluation protocol, and the implementation for six state-of-the-art layout-guided diffusion models.

## Abstract

Evaluating layout-guided text-to-image generative models requires assessing both semantic alignment with textual prompts and spatial fidelity to prescribed layouts. Assessing layout alignment requires collecting fine-grained annotations, which is costly and labor-intensive. Consequently, current benchmarks rarely provide comprehensive layout evaluation and often remain limited in scale or coverage, making model comparison, ranking, and interpretation difficult. In this work, we introduce a closed-set benchmark (C-Bench) designed to isolate key generative capabilities while providing varying levels of complexity in both prompt structure and layout. To complement this controlled setting, we propose an open-set benchmark (O-Bench) that evaluates models using real-world prompts and layouts, offering a measure of semantic and spatial alignment in the wild. We further develop a unified evaluation protocol that combines semantic and spatial accuracy into a single score, ensuring consistent model ranking. Using our benchmarks, we conduct a large-scale evaluation of six state-of-the-art layout-guided diffusion models, totaling 319,086 generated and evaluated images. We establish a model ranking based on their overall performance and provide detailed breakdowns for text and layout alignment to enhance interpretability. Fine-grained analyses across scenarios and prompt complexities highlight the strengths and limitations of current models. Code is available at https://github.com/lparolari/cobench.

## Project Structure

```
.
├── benchmarks
│   ├── instructions
│   └── notebooks
├── eval
└── zoo
    ├── attention-refocusing
    ├── boxdiff
    ├── gligen
    ├── layout-guidance
    ├── MIGC
    └── sd14
```



## Benchmarks

Benchmarks are available in the directory `benchmarks/instructions`. It includes:

* **Closed-Set Benchmark (C-Bench)**
  Automatically generated using a combination of template-based and LLM-based instructions with constraint-driven layouts.
  Designed to isolate generative capabilities under controlled conditions.

* **Open-Set Benchmark (O-Bench)**
  Derived from Flickr30k Entities to evaluate models on natural prompts and real-world layouts.
  Supports analysis of generalization in unconstrained settings.

The directory `notebooks` contains scripts used to generate our benchmarks. You can modify them to obtain your own version of the benchmark or to scale them to a higher number of examples.

* `generate_obench.ipynb`: Generate O-Bench from Flickr30k test split.
* `generate_cbench.ipynb`: Generate C-Bench from template-based rules. It includes the prompts used to generate instructions with ChatGPT (Note: To avoid high costs, the script does not generate automatically the instructions. We obtained *complex composition* instructions by prompting ChatGPT via the web interface and manually collected the results).
* `aggregate.ipynb`: Aggregate the results of the evaluation of a model by scenarios and by number of objects.
* `visualize_bboxes.ipynb`: Inspect and visualize instructions from a benchmark.

## Evaluation

We provide a unified evaluation protocol that combines **semantic alignment** with **spatial fidelity** into a single score. This allows for:

* Consistent ranking of models
* Detailed interpretability of strengths and weaknesses
* Reproducible benchmarking

Example evaluation scripts are available in `eval/tifa_test.py`. Dependencies are listed in `eval/requirements.txt`.

## Model Zoo

This repository includes pre-configured setups for several layout-guided diffusion models:

* `attention-refocusing`
* `boxdiff`
* `gligen`
* `layout-guidance`
* `MIGC`
* `sd14`

Each folder contains the necessary scripts and checkpoints for generating images from layout-guided instructions.

## Getting Started

Both evaluation an zoo are self documented and include a readme to easily setup both the evalation protocol or generate the images through layout-guided diffusion models.

## Citation

If you use this repository in your research, please cite our paper:

```
COMING SOON in the proceedings of CVPR Findings
```

