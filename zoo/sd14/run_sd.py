import time
import torch
import os
import pandas as pd

from diffusers import StableDiffusionPipeline
from utils import logger
import torchvision.utils
import torchvision.transforms.functional as tf

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

# Generate an image described by the prompt and
# insert objects described by text at the region defined by bounding boxes
def main():
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",use_safetensors=False, safety_checker=None)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    pipe = pipe.to(device)

    height=512
    width=512
    seeds = range(1,17)

    #bench=make_tinyHRS()
    bench=readPromptsCSV(os.path.join("prompts","prompt_collection_bboxes.csv"))

    model_name="PromptCollection-SD14"
    
    if (not os.path.isdir("./results/"+model_name)):
            os.makedirs("./results/"+model_name)
    
    #intialize logger
    l=logger.Logger("./results/"+model_name+"/")
    
    # ids to iterate the dict
    ids = []
    for i in range(0,len(bench)):
        ids.append(str(i).zfill(3))
        
    for id in ids:
        
        output_path = "./results/"+model_name+"/"+ id +'_'+bench[id]['prompt'] + "/"

        if (not os.path.isdir(output_path)):
            os.makedirs(output_path)

        print("Sample number ",id)
        
        torch.cuda.empty_cache()

        gen_images=[]
        gen_bboxes_images=[]

        for seed in seeds:
            print(f"Current seed is : {seed}")

            # start stopwatch
            start = time.time()

            if torch.cuda.is_available():
                g = torch.Generator('cuda').manual_seed(seed)
            else:
                g = torch.Generator('cpu').manual_seed(seed)

            images = pipe(
                prompt=bench[id]['prompt'],
                height=height,
                width=width,
                output_type="pil",
                num_inference_steps=50,
                generator=g,
                negative_prompt='low quality, low res, distortion, watermark, monochrome, cropped, mutation, bad anatomy, collage, border, tiled').images

            # end stopwatch
            end = time.time()
            # save to logger
            l.log_time_run(start, end)

            image=images[0]

            image.save(output_path +"/"+ str(seed) + ".jpg")
            gen_images.append(tf.pil_to_tensor(image))

        # save a grid of results across all seeds without bboxes
        tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_images,nrow=4,padding=0)).save(output_path +"/"+ bench[id]['prompt'] + ".png")
    
    # log gpu stats
    l.log_gpu_memory_instance()
    # save to csv_file
    l.save_log_to_csv(model_name)
    print("End of generation process for ", model_name)
        
if __name__ == '__main__':
    main()
