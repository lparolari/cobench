# Testing StableDiffusion 1.4 for my master thesis

This repo uses Stable Diffusion 1.4 for text-to-image generation. It does not include layout guidance or bounding box constraints — images are generated freely based on the prompt.

## Quick start

```bash
conda create --name sd python=3.10
conda activate sd
pip install -r requirements.txt
```

## Image generation

The .csv file containing the prompts should be inside a folder named `prompts` that is posiotioned in the root of the project.

The .csv file used is expected to have the following structure (no limits in the number of objects):
`id,prompt,obj1,bbox1,obj2,bbox2,obj3,bbox3,obj4,bbox4`

While Stable Diffusion 1.4 does not use bounding box data for image generation, the input CSV should still contain bounding box fields. This ensures compatibility with other models discussed in my thesis and maintains a consistent data format across experiments.
