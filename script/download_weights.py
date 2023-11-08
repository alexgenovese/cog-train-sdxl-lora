import os, torch, time, shutil
from diffusers import AutoencoderKL, DiffusionPipeline, StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import ( StableDiffusionSafetyChecker )
from transformers import (Blip2Processor, CLIPSegProcessor, Swin2SRForImageSuperResolution)

SDXL_MODEL_CACHE = "./sdxl-cache"
VAE_CACHE = "./vae-cache"

# VAE CACHE CHECKER
# if not os.path.exists(VAE_CACHE):
#    better_vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
#    better_vae.save_pretrained(VAE_CACHE, safe_serialization=True)

better_vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)

# SDXL CACHE CHECKER
if not os.path.exists(SDXL_MODEL_CACHE):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=better_vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.save_pretrained(SDXL_MODEL_CACHE, safe_serialization=True)


# Download the preprocess models
BLIP_REPO_ID = "Salesforce/blip-image-captioning-large"
BLIP_FOLDER = "/Blip"
CIDAS_REPO_ID = "CIDAS/clipseg-rd64-refined"
CIDAS_FOLDER = "/CIDAS"
UPSCALER_REPO_ID = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
UPSCALER_FOLDER = "/Caidas"
PREPROCESS_MODEL_CACHE = "./preprocess-cache"

# Check folder exists
if not os.path.exists(PREPROCESS_MODEL_CACHE+UPSCALER_FOLDER):
    upscaler = Swin2SRForImageSuperResolution.from_pretrained(UPSCALER_REPO_ID, cache_dir=PREPROCESS_MODEL_CACHE)
    upscaler.save_pretrained(PREPROCESS_MODEL_CACHE+UPSCALER_FOLDER, safe_serialization=True)

if not os.path.exists(PREPROCESS_MODEL_CACHE+CIDAS_FOLDER):
    cidas = CLIPSegProcessor.from_pretrained(CIDAS_REPO_ID, cache_dir=PREPROCESS_MODEL_CACHE)
    cidas.save_pretrained(PREPROCESS_MODEL_CACHE+CIDAS_FOLDER, safe_serialization=True)

if not os.path.exists(PREPROCESS_MODEL_CACHE+BLIP_FOLDER):
    blip2 = Blip2Processor.from_pretrained(BLIP_REPO_ID, cache_dir=PREPROCESS_MODEL_CACHE)
    blip2.save_pretrained(PREPROCESS_MODEL_CACHE+BLIP_FOLDER, safe_serialization=True)


BASE_MODEL = "alexgenovese/reica06"
BASE_MODEL_CACHE = './weights-cache/base_model'


def cache_base_model():
    if not os.path.exists(BASE_MODEL_CACHE):
        try:
            os.makedirs(BASE_MODEL_CACHE)
            start = time.time()
            print("Converting Reica custom model")
            pipe = StableDiffusionXLPipeline.from_pretrained( BASE_MODEL )
            pipe.save_pretrained(BASE_MODEL_CACHE, safe_serialization=True)
            print("Downloading took: ", time.time() - start)
        except Exception as error:
            print("BASE_MODEL - Something went wrong while downloading")
            print(f"{error}")
            shutil.rmtree(BASE_MODEL_CACHE)
            print("BASE_MODEL - Removed empty cache directory")