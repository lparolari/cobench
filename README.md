# Layout-Guided Text-to-Image Generation Benchmarks and Evaluation

This repository provides the benchmarks, evaluation tools, and model zoo used in our study on layout-guided text-to-image generative models. It includes both **closed-set** and **open-set** benchmarks, a unified evaluation protocol, and implementations for state-of-the-art layout-guided diffusion models.

## Abstract

Evaluating layout-guided text-to-image generative models requires measuring both semantic alignment with textual prompts and spatial fidelity to prescribed layouts. Existing benchmarks are limited in scale and coverage, hindering systematic comparison and reducing interpretability of model capabilities. In this paper, we introduce a scalable closed-set benchmark (C-Bench), automatically built through a pipeline combining template- and LLM-based prompt generation with constraint-driven layout synthesis. C-Bench spans seven scenarios designed to isolate key generative capabilities and provides varying levels of complexity in both prompt structure and layout. To complement this controlled setting, we propose an open-set benchmark (O-Bench) derived from Flickr30k Entities, enabling evaluation on natural prompts and layouts. We further develop a unified evaluation protocol that combines semantic and spatial accuracy into a single score, enabling consistent model ranking. Using our benchmarks, we conduct a large-scale evaluation of six state-of-the-art layout-guided diffusion models, totaling 319,086 generated and evaluated images. Results show that MIGC achieves the highest overall performance (0.7082 on C-Bench and 0.7548 on O-Bench), establishing it as the most reliable model, particularly in layout alignment. Models trained explicitly with layout information consistently outperform Stable Diffusion–based approaches, which lag significantly behind. Overall, our benchmarks and evaluation protocol provide a scalable and interpretable framework for assessing progress in controllable image generation. Code and benchmarks will be released upon acceptance.

## Project Structure

```
.
├── benchmarks
├── eval
└── zoo
    ├── attention-refocusing
    ├── boxdiff
    ├── gligen
    ├── layout-guidance
    ├── MIGC
    └── sd14
```

---

## Benchmarks

* **Closed-Set Benchmark (\csb{})**
  Automatically generated using a combination of template-based and LLM-based prompts with constraint-driven layouts.
  Designed to isolate generative capabilities under controlled conditions.

* **Open-Set Benchmark (\osb{})**
  Derived from Flickr30k Entities to evaluate models on natural prompts and real-world layouts.
  Supports analysis of generalization in unconstrained settings.

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

Each folder contains the necessary scripts and checkpoints for generating images from layout-guided prompts.

## Getting Started

Both evaluation an zoo are self documented and include a readme to easily setup both the evalation protocol or generate the images through layout-guided diffusion models.

## Citation

If you use this repository in your research, please cite our paper:

```
TBD: under double-blind review

@inproceedings{yourpaper2025,
  title={Title of Your Paper},
  author={Your Name et al.},
  booktitle={Conference Name},
  year={2025}
}
```

