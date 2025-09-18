# Scalable Evaluation of Closed-Set and Open-Set Semantic and Spatial Alignment in Layout-Guided Diffusion Models

This repository contains the additional material developed for my matser thesis. 

## Content structure
The repopsitory content is organized as follows:

- `prompts`: contains the prompt collections generated for the evaluation of layout-guided diffusion models. In particular:
    - `c_bench.csv`: is the prompt collection used for the closed-set benchmark;
    - `o_bench.csv`: is the prompt collection used for the open-set benchmark.
- `notebooks`: contains additional notebooks developed.In particular:
    - `flickr30k.ipynb`: code used to downsample Flickr30k test split to collect prompts and bboxes for the open-set benchmark;
    - `randomPromptsAndBoxes.ipynb`: code used to generate automatically prompts and bounding boxes;
    - `results_avg.ipynb`: code used to aggregate the results of the evaluation of a model by scenarios and by number of objects;
    - `visualize_bboxes.ipynb`: code used to visualize bboxes randomly or by Id.
