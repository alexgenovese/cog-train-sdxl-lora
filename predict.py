import os, shutil, tarfile, time, subprocess, torch
from tqdm import tqdm
from cog import BasePredictor, Input, Path
from typing import Tuple
# Before starting the training process
from preprocess import preprocess
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
# Class for training LoRA
from trainer_pti import main_trainer
from huggingface_hub import login

# Defining Static Variables
FEATURE_EXTRACTOR = "./feature-extractor"
OUTPUT_DIR = "training_out"
HF_TOKEN = "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"
BASE_MODEL = "SG161222/RealVisXL_V2.0"
BASE_MODEL_CACHE = "./base-model-cache"

# Global functions
## Download the models if not present in cache folder yet
def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", url, dest])
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):

    def setup(self): 
        start = time.time()
        print("Setup started...")
        # Login in HF
        login( token = HF_TOKEN )
        
        with tqdm(total=100, desc="Setup") as pbar:
            self.device = self.get_device_type()
            self.torch_type = self.get_torch_type()
            self.variant = "fp32" if torch.is_floating_point(torch.tensor(32)) else "fp16"
            pbar.update(10)

            print(f"Settings {self.variant} {self.torch_type} {self.device}")

            # Create SDXL 
            if not os.path.exists(BASE_MODEL_CACHE):
                self.in_base_model = DiffusionPipeline.from_pretrained( BASE_MODEL, torch_dtype=self.torch_type )
                self.in_base_model.save_pretrained(BASE_MODEL_CACHE)
            pbar.update(10)
        
        print("setup took: ", time.time() - start)


    def get_torch_type(self):
        if torch.backends.mps.is_available():
            print("Torch MPS")
            return torch.float32
        
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16

            if torch.is_floating_point(torch.tensor(32)):
                return torch.float32
            
            if torch.is_floating_point(torch.tensor(16)):
                return torch.float16

        return torch.float16

    def get_device_type(self):
        if torch.backends.mps.is_available():
            print("MPS Device Type")
            return "mps"
        
        if torch.cuda.is_available():
            print("CUDA Device type")
            return "cuda"
        
        print("CPU Device type")
        return "cpu"
        

    # working locally
    def predict(
        self,
        input_images: str = "./_training_dataset/geox.zip",
        seed: int = 123456,
        resolution: int = 1024,
        train_batch_size: int = 3,
        num_train_epochs: int = 20,
        max_train_steps: int = None,
        class_token: str = "bag",
        is_lora: bool = True,
        unet_learning_rate: float = 1e-04,
        ti_lr: float = 1e-04, # 0,0001
        lora_lr: float = 4e-4, # 0,0004
        lora_rank: int = 32,
        lora_rank_alpha: int = 16,
        lr_scheduler: str = "constant",
        optimization: str = "AdamW",
        lr_warmup_steps: int = 0,
        token_string: str = "siduhc",
        caption_prefix: str = "a photo of siduhc shoes",
        mask_target_prompts: str = None,
        crop_based_on_salience: bool = True,
        clipseg_temperature: float = 1.0,
        verbose: bool = True,
        checkpointing_steps: int = 999999,
        input_images_filetype: str = "infer",
    ) -> Path:
        # Hard-code token_map for now. Make it configurable once we support multiple concepts or user-uploaded caption csv.
        token_map = token_string + ":2"

        # Process 'token_to_train' and 'input_data_tar_or_zip'
        inserting_list_tokens = token_map.split(",")

        token_dict = {}
        running_tok_cnt = 0
        all_token_lists = []
        for token in inserting_list_tokens:
            n_tok = int(token.split(":")[1])

            token_dict[token.split(":")[0]] = "".join(
                [f"<s{i + running_tok_cnt}>" for i in range(n_tok)]
            )
            all_token_lists.extend([f"<s{i + running_tok_cnt}>" for i in range(n_tok)])

            running_tok_cnt += n_tok
       
        # Prepare for training
        input_dir = preprocess(
            input_images_filetype=input_images_filetype,
            input_zip_path=input_images,
            caption_text=caption_prefix,
            mask_target_prompts=mask_target_prompts,
            target_size=resolution,
            crop_based_on_salience=crop_based_on_salience,
            use_face_detection_instead=False,
            temp=clipseg_temperature,
            substitution_tokens=list(token_dict.keys()),
            class_token=class_token,
            device=self.device,
            dtype=self.torch_type
        )

        # Clean and create output directory
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)


        main_trainer(
            pretrained_model_name_or_path=BASE_MODEL_CACHE,
            instance_data_dir=os.path.join(input_dir, "captions.csv"),
            output_dir=OUTPUT_DIR,
            seed=seed,
            resolution=resolution,
            train_batch_size=train_batch_size,
            num_train_epochs=num_train_epochs,
            max_train_steps=max_train_steps,
            gradient_accumulation_steps=1,
            unet_learning_rate=unet_learning_rate,
            ti_lr=ti_lr,
            lora_lr=lora_lr,
            lr_scheduler=lr_scheduler,
            lr_warmup_steps=lr_warmup_steps,
            token_dict=token_dict,
            inserting_list_tokens=all_token_lists,
            verbose=verbose,
            checkpointing_steps=checkpointing_steps,
            scale_lr=False,
            max_grad_norm=1.0,
            allow_tf32=True,
            mixed_precision=self.torch_type,
            device=self.device,
            lora_rank=lora_rank,
            lora_rank_alpha=lora_rank_alpha,
            is_lora=is_lora,
        )
        

        directory = Path(OUTPUT_DIR)
        out_path = "trained_model.tar"
        with tarfile.open(out_path, "w") as tar:
            for file_path in directory.rglob("*"):
                print(file_path)
                arcname = file_path.relative_to(directory)
                tar.add(file_path, arcname=arcname)


        return Path(out_path)
        # return out_path
    

def main():
    pred = Predictor()
    pred.setup()
    pred.predict()


if __name__ == "__main__":
    main()