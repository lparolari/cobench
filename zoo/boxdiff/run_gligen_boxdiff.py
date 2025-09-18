import os
import pprint
import time
from typing import List
import pandas as pd
import math

#import pyrallis
import torch
from PIL import Image
from config import RunConfig
from pipeline.gligen_pipeline_boxdiff import BoxDiffPipeline
from utils import ptp_utils, logger
from utils.ptp_utils import AttentionStore

import torchvision.utils
import torchvision.transforms.functional as tf

import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def readPromptsCSV(path):
    df = pd.read_csv(path, dtype={'id': str})
    conversion_dict = {}
    for i in range(len(df)):
        entry = {'prompt': df.at[i, 'prompt']}
        # Dynamically find all obj/bbox columns and keep original naming
        for col in df.columns:
            if col.startswith('obj'):
                idx = col[3:]
                bbox_col = f'bbox{idx}'
                obj_val = df.at[i, col]
                bbox_val = df.at[i, bbox_col] if bbox_col in df.columns else None
                # Always include obj and bbox, even if NaN for retro compatibility
                entry[col] = obj_val
                if bbox_col in df.columns:
                    entry[bbox_col] = bbox_val
        conversion_dict[df.at[i, 'id']] = entry
    return conversion_dict

def load_model():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    stable_diffusion_version = "masterful/gligen-1-4-generation-text-box"
    stable = BoxDiffPipeline.from_pretrained(stable_diffusion_version, use_safetensors=False,local_files_only=True).to(device)

    return stable

def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices 

def assign_indices_to_alter(stable, prompt: str,gligen_phrase:List[str] ) -> List[int]:
    print(stable.tokenizer(prompt)['input_ids'])
    input()
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices

def calculateTokenIndices(stable, prompt: str, phrases) -> List[int]:
    token_indices = []
    tokenized = stable.tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]
    token_idx_to_word = {
        idx: stable.tokenizer.decode([token_id])
        for idx, token_id in enumerate(input_ids)
        if 0 < idx < len(input_ids) - 1  # skip start/end tokens
    }

    print("Prompt tokens:")
    for key, value in token_idx_to_word.items():
        print(f"{key}: '{value}'")

    for target_token in phrases:
        max_len = 0
        candidate_token_key = None
        for key, value in token_idx_to_word.items():
            if value.lower() in target_token.lower():
                if len(value) > max_len:
                    max_len = len(value)
                    candidate_token_key = key
        if candidate_token_key is not None:
            token_indices.append(candidate_token_key)
        else:
            raise ValueError(f"No matching token found for target: '{target_token}'")

    return token_indices

def run_on_prompt(prompt: List[str],
                  model: BoxDiffPipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:

    if controller is not None:
        ptp_utils.register_attention_control(model, controller)

    gligen_boxes = []
    for i in range(len(config.bboxes)):
        x1, y1, x2, y2 = config.bboxes[i]
        gligen_boxes.append([x1/512, y1/512, x2/512, y2/512])

    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    gligen_phrases=config.gligen_phrases,
                    gligen_boxes=gligen_boxes,
                    gligen_scheduled_sampling_beta=0.3,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    bbox=config.bboxes,
                    height=512,
                    width=512,
                    config=config,
                    negative_prompt='low quality, low res, distortion, watermark, monochrome, cropped, mutation, bad anatomy, collage, border, tiled'
                    )
    image = outputs.images[0]
    return image


#@pyrallis.wrap()
def main(config: RunConfig):

    stable = load_model()
    #token_indices = get_indices_to_alter(stable, config.prompt, config.gligen_phrases) if config.token_indices is None else config.token_indices
    token_indices = calculateTokenIndices(stable, config.prompt, config.gligen_phrases)
    
    gen_images = []
    gen_bboxes_images=[]
    for seed in config.seeds:
        print(f"Current seed is : {seed}")

        #start stopwatch
        start=time.time()

        if torch.cuda.is_available():
            g = torch.Generator('cuda').manual_seed(seed)
        else:
            g = torch.Generator('cpu').manual_seed(seed)

        controller = AttentionStore()
        controller.num_uncond_att_layers = -16
        image = run_on_prompt(prompt=config.prompt,
                              model=stable,
                              controller=controller,
                              token_indices=token_indices,
                              seed=g,
                              config=config)
        #end stopwatch
        end = time.time()
        #save to logger
        l.log_time_run(start,end)


        #image.save(prompt_output_path / f'{seed}.png')
        image.save(str(config.output_path) +"/"+ str(seed) + ".jpg")
        #list of tensors
        gen_images.append(tf.pil_to_tensor(image))

        
        #draw the bounding boxes
        image=torchvision.utils.draw_bounding_boxes(tf.pil_to_tensor(image),
                                                    torch.Tensor(config.bboxes),
                                                    labels=config.gligen_phrases,
                                                    colors=['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green'],
                                                    width=4,
                                                    font="font.ttf",
                                                    font_size=20)
        #list of tensors
        gen_bboxes_images.append(image)
        tf.to_pil_image(image).save(output_path+str(seed)+"_bboxes.png")

    # save a grid of results across all seeds without bboxes
    tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_images,nrow=4,padding=0)).save(str(config.output_path) +"/"+ config.prompt + ".png")

    # save a grid of results across all seeds with bboxes
    tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_bboxes_images,nrow=4,padding=0)).save(str(config.output_path) +"/"+ config.prompt + "_bboxes.png")


if __name__ == '__main__':
    height = 512
    width = 512
    seeds = range(1,9)

    bench=readPromptsCSV(os.path.join("prompts","openSet.csv"))

    model_name="openSet-G_BD"
    
    if (not os.path.isdir("./results/"+model_name)):
            os.makedirs("./results/"+model_name)
    
    #intialize logger
    l=logger.Logger("./results/"+model_name+"/")
    
    ids = []
    for i in range(0,len(bench)):
        ids.append(str(i).zfill(4))

    for id in ids:
        bboxes = []
        phrases = []
        # Dynamically find all obj/bbox columns for each id
        for col in bench[id]:
            if col.startswith('obj') and bench[id][col] is not None and not (isinstance(bench[id][col], (int, float)) and math.isnan(bench[id][col])):
                idx = col[3:]  # get the number after 'obj'
                bbox_col = f'bbox{idx}'
                if bbox_col in bench[id] and bench[id][bbox_col] is not None:
                    phrases.append(bench[id][col])
                    bboxes.append([int(x) for x in bench[id][bbox_col].split(',')])
            
        output_path = "./results/"+model_name+"/"+ id +'_'+bench[id]['prompt'] + "/"

        if (not os.path.isdir(output_path)):
            os.makedirs(output_path)
        print("Sample number ",id)
        torch.cuda.empty_cache()
        
        main(RunConfig(
            prompt_id=id,
            prompt=bench[id]['prompt'],
            gligen_phrases=phrases,
            seeds=seeds,
            bboxes=bboxes,
            output_path=output_path,
        )) 
    
    #log gpu stats
    l.log_gpu_memory_instance()
    #save to csv_file
    l.save_log_to_csv(model_name)
    print("End of generation process for ", model_name)

