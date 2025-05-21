import logging
import datetime
import yaml  # Add yaml import
# import tempfile # Removed - Unused
# import csv # Removed - Unused
# import shutil # Removed - Unused
import math
import torch.nn.functional as F
# from speechbrain.processing.signal_processing import  reverberate # Removed - Unused
from datasets import load_dataset, Audio, DatasetDict, concatenate_datasets # Ensure all are here from datasets
import torchaudio # Keep for audio processing if needed elsewhere, e.g. resampling in HF Datasets

# Turn off tokenizers parallelism warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Cell 1: Basic Modal Setup
from speechbrain.augment.time_domain import DropChunk, DropFreq, DropBitResolution # Actively used
from speechbrain.augment.augmenter import Augmenter # Actively used
import random # For augmentation logic

import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch # Should already be imported by SB or other components, but good to ensure

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer # Added for HF Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting script execution...")

logging.info("Importing system libraries...")
import sys
import os
logging.info("System libraries imported successfully")

logging.info("Importing modal...")
import modal
logging.info("Modal imported successfully")

logging.info("Importing transformers...")
from transformers import WhisperProcessor
logging.info("Transformers import completed successfully")

# --- Basic Logging Setup ---
logging.info("Configuring logging system...")

# --- Verify Whisper Token IDs ---
logging.info("Defining verify_whisper_tokens function...")
def verify_whisper_tokens(model_name="openai/whisper-large-v3-turbo", language="ar", task="transcribe"): # Placeholder : Change this if you want to use a different whisper model
    logging.info(f"Verifying Whisper tokens for model {model_name}")
    try:
        # Use WhisperProcessor to load both tokenizer and feature extractor
        processor = WhisperProcessor.from_pretrained(model_name) # Load base processor first
        
        # Set language and task AFTER loading - this configures the tokenizer inside the processor
        processor.tokenizer.set_prefix_tokens(language=language, task=task)

        tokenizer = processor.tokenizer # Get the configured tokenizer

        # Now access the IDs from the configured tokenizer
        sot_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        eot_id = tokenizer.eos_token_id # Typically <|endoftext|>
        pad_id = tokenizer.pad_token_id # Often the same as EOS
        lang_token_id = tokenizer.convert_tokens_to_ids(f"<|{language}|>")
        task_token_id = tokenizer.convert_tokens_to_ids(f"<|{task}|>")
        no_timestamps_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>") # Often needed

        logging.info(f"SOT ID (<|startoftranscript|>): {sot_id}")
        logging.info(f"EOT ID (<|endoftext|> / EOS): {eot_id}")
        logging.info(f"PAD ID: {pad_id}")
        logging.info(f"Language Token ID (<|{language}|>): {lang_token_id}")
        logging.info(f"Task Token ID (<|{task}|>): {task_token_id}")
        logging.info(f"No Timestamps Token ID (<|notimestamps|>): {no_timestamps_id}")

        # Recommended decoder prefix includes SOT, lang, task, and often <|notimestamps|> for transcription
        decoder_start_ids = [sot_id, lang_token_id, task_token_id, no_timestamps_id]
        
        logging.info(f"Recommended decoder start IDs: {decoder_start_ids}")
        logging.info(f"Recommended target end ID (EOS): {eot_id}")

        return {
            "sot_token_id": sot_id,
            "eos_token_id": eot_id,
            "pad_token_id": pad_id,
            "lang_token_id": lang_token_id,
            "task_token_id": task_token_id,
            "no_timestamps_id": no_timestamps_id,
            "decoder_start_ids": decoder_start_ids
        }
    except Exception as e:
        logging.error(f"Error verifying Whisper tokens: {e}", exc_info=True)
        return None

print("Defining Modal App basics...")

# --- Modal App Setup ---
app = modal.App("fine-tune-whisper-large-egyptian-arabic") # Placeholder : name your modal app here 

# --- Secrets ---
hf_secret = modal.Secret.from_name("huggingface-secret-write") # Define the huggingface secret object
wandb_secret = modal.Secret.from_name("wandb-secret") # Define the wandb secret object

# --- Persistent Storage ---
volume = modal.Volume.from_name(
    "fine-tune-whisper-large-egyptian-arabic", create_if_missing=True # Placeholder : name your volume here 
)
CACHE_DIR = "/cache" # HF cache inside container
CHECKPOINT_DIR = "/root/checkpoints" # Mount point inside container

# --- Environment Image Definition ---
modal_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "build-essential", "cmake", "libboost-all-dev",
        "libeigen3-dev", "git", "libsndfile1", "ffmpeg",
        "wget" 
    )
    .pip_install(
        # Existing user packages (kept if not conflicting or outdated for Unsloth)
        "pip==23.3.2",
        "setuptools==69.0.3",
        "wheel==0.42.0",
        "pyarrow==15.0.0",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        "transformers==4.51.3", # User's version, compatible with trl 0.15.2
        "wandb",
        "speechbrain==1.0.3",  # Kept for augmentations
        "librosa==0.10.1",
        "huggingface_hub==0.30.0", # User's version
        "sentencepiece==0.1.99", # User's version
        "num2words==0.5.13",
        "pyyaml==6.0.1",
        "tqdm==4.66.1",
        "pandas==2.1.4",
        "soundfile==0.12.1",
        "flash-attn==2.4.2", 
        "bitsandbytes==0.45.4", # User's version

        # Updates and Unsloth additions
        "accelerate>=0.28.0",  # Updated (trl 0.15.2 needs >=0.28.0)
        "datasets>=3.4.1",    # Updated (as per Unsloth's Colab example)
        "unsloth",
        "xformers==0.0.29.post3", # From Unsloth Colab example
        "peft>=0.12.0",          # Current recommended version
        "trl==0.15.2",           # From Unsloth Colab example
        "evaluate",
        "jiwer",
    )
    # Download noise & RIR WAVs during image build, you will want to change this to increase the number of files 
    .run_commands(
        "mkdir -p /noise_assets /rir_assets", # Create both directories
        # Noise files
        "wget https://www.dropbox.com/s/aleer424jumcs08/noise2.wav -O /noise_assets/noise2.wav",
        "wget https://www.dropbox.com/s/eoxxi2ezr8owk8a/noise3.wav -O /noise_assets/noise3.wav",
        # RIR file
        "wget https://www.dropbox.com/s/pjnub2s5hql2vxs/rir1.wav -O /rir_assets/rir1.wav"
    )
)

print("Modal App basics defined (using modal.Volume.from_name).")
# End of Cell 1
 
# Ensure persistent Hugging Face cache inside the volume
CHECKPOINT_DIR = "/root/checkpoints"  # Mount point inside container (defined above)
CACHE_DIR = f"{CHECKPOINT_DIR}/hf_cache"  # Persistent HF cache stored in Modal volume

# Create the cache directory if it doesn't exist yet
os.makedirs(CACHE_DIR, exist_ok=True)

# Redirect common HF cache environment variables to this persistent location
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR

# Cell 2: Define Hyperparameters (Updated with correct token IDs)

print("Defining Hyperparameters...")

# Verify token IDs first
token_info = verify_whisper_tokens(model_name="openai/whisper-large-v3-turbo", language="ar", task="transcribe") #Placeholder : Change this if you want to use a different whisper model

TARGET_SAMPLE_RATE = 16000 # Define target sample rate globally

hparams = {
    # Data
    "hf_dataset_id": "MAdel121/arabic-egy-cleaned", # Placeholder : Change this if you want to use a different dataset
    "train_split": "train",
    "valid_split": "validation",
    "test_split": "test",
    "target_sample_rate": TARGET_SAMPLE_RATE,

    # Model & Tokenizer
    "whisper_hub": "openai/whisper-large-v3-turbo", # Placeholder : Change this if you want to use a different whisper model
    "save_folder": f"{CHECKPOINT_DIR}/whisper_large_egy_save",
    "output_folder": f"{CHECKPOINT_DIR}/whisper_large_egy_output",
    "language": "ar",
    "task": "transcribe",

    # Token IDs (Updated from verification using Processor)
    "sot_index": token_info["sot_token_id"] if token_info else 50258,
    "eos_index": token_info["eos_token_id"] if token_info else 50257,
    "pad_token_id": token_info["pad_token_id"] if token_info else 50257,
    "decoder_start_ids": token_info["decoder_start_ids"] if token_info else [50258, 50361, 50359, 50363],

    # HF Dataset Preprocessing hparams
    "text_column_name": "text", # Default column name for transcriptions
    "preprocessing_batch_size": 16, # Batch size for dataset.map()
    "num_map_workers": None, # Default to os.cpu_count(), set to 1 if issues arise

    # === Augmentation Configuration ===
    "augment": True, # Master switch for augmentation
    "augment_prob_master": 0.50, # Overall probability the Augmenter chain is applied per batch

    # --- Boolean flags to enable specific augmentations ---
    "use_add_noise": False,          # Set to True to enable AddNoise
    "use_add_reverb": False,         # Set to True to enable AddReverb
    "use_speed_perturb": False,      # Set to True to enable SpeedPerturb
    "use_pitch_shift": False,        # Set to True to enable PitchShiftWrapper
    "use_gain": False,               # Set to True to enable GainWrapper
    "use_drop_chunk": True,          # Set to True to enable DropChunk 
    "use_drop_freq": True,           # Set to True to enable DropFreq 
    "use_do_clip": False,            # Set to True to enable DoClip
    "use_drop_bit_resolution": True, # Set to True to enable DropBitResolution 
    "use_codec_augment": False,      # Set to True to enable CodecAugment

    # --- Parameters for *enabled* augmentations ---
    "min_augmentations": 1, # Min number of augmentations to apply from the enabled pool
    "max_augmentations": 3, # Max number of augmentations to apply from the enabled pool

    # AddNoise Params (Used if use_add_noise is True)
    "noise_snr_low": 15,
    "noise_snr_high": 25,

    # AddReverb Params (Used if use_add_reverb is True)
    "rir_assets_dir": "/rir_assets",
    "rir_manifest_path": "/tmp/rir_manifest.csv",
    "rir_scale_factor": 1.0,

    # SpeedPerturb Params (Used if use_speed_perturb is True)
    "speed_factors": [95, 105],

    # PitchShiftWrapper Params (Used if use_pitch_shift is True)
    "pitch_steps_low": -1,
    "pitch_steps_high": 1,

    # GainWrapper Params (Used if use_gain is True)
    "gain_db_low": -4,
    "gain_db_high": 4,

    # DropChunk Params (Used if use_drop_chunk is True)
    "drop_chunk_length_low": 1600,
    "drop_chunk_length_high": 4800,
    "drop_chunk_count_low": 1,
    "drop_chunk_count_high": 5,

    # DropFreq Params (Used if use_drop_freq is True)
    "drop_freq_count_low": 1,
    "drop_freq_count_high": 3,

    # DoClip Params (Used if use_do_clip is True)
    "clip_low": 0.7,
    "clip_high": 0.9,

    # === End Augmentation Configuration ===

    # Training Params
    "seed": 1986,
    "epochs": 10,
    "learning_rate": 1e-5, # Probably a bit high, need to decrease
    "optimizer_type": "AdamW", # Added for tracking
    "scheduler_type": "NewBob", # Added for tracking
    "lr_improvement_threshold": 0.0025, # Default for NewBobScheduler 
    "lr_patient": 0, # Default for NewBobScheduler 
    "lr_warmup_steps": 1000,
    "weight_decay": 0.05,
    "lr_annealing_factor": 0.9, # Used by NewBobScheduler
    "batch_size_dynamic": False, # DISABLED DYNAMIC BATCHING
    "dynamic_batch_num_buckets": 60, # (Not used when dynamic batching is False)
    "loader_batch_size": 12, # Used only if batch_size_dynamic is False
    "max_batch_len_seconds": 40.0,
    "num_workers": 8,
    "grad_accumulation_factor": 2,
    "max_grad_norm": 5.0,

    # Checkpointing
    "ckpt_interval_minutes": 30,
    "num_checkpoints_to_keep": 2,

    # Whisper decoding params
    "num_beams": 5,

    # === W&B Configuration ===
    "use_wandb": False,  # Flag to enable/disable weights and biases
    "wandb_project": "whisper-large-egyptian-arabic",  # Placeholder : Project name on weights and biases 
    "wandb_entity": None,  # Placeholder - Optional: Your W&B username or team name
    "wandb_log_batch_freq": 100,  # Log batch metrics every N steps
    "wandb_watch_model": True,  # Whether to watch gradients
    "wandb_watch_freq": 100,  # How often to log gradients
    "wandb_resume_id": None, # Set to specific run ID string to resume a  W & B run
}

# Add a check after hparams definition
if not token_info:
    logging.warning("Failed to verify token IDs. Using default values in hparams, which might be incorrect.")
else:
    logging.info("Successfully verified token IDs and updated hparams.")

print("Hyperparameters defined.")
# End of Cell 2


# Cell 3: Define Data Loading/Preprocessing 

import torch # Already imported earlier
# import torchaudio # Already imported globally
# import random # Already imported earlier
import speechbrain as sb # Used for sb.utils.seed_everything
# from speechbrain.utils.data_pipeline import takes, provides # Removed - Unused

# print("Defining Data Loading Pipelines...") # Removed

# audio_pipeline_minimal and text_pipeline_minimal functions fully removed.

# print("Data Loading Pipelines defined.") # Commented out as pipelines are removed
# End of Cell 3


# Cell 3.5: Define Data Collator (Global Scope)
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any # Type hint for WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths
        # and need different padding methods.
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# Cell 4: Define Brain Subclass - THIS CELL IS NOW REMOVED
# (Content of WhisperFineTuneBrain and its helper classes, including PitchShiftWrapper, GainWrapper, RIRSampler, were removed in Turn 11)
# All related imports like ErrorRateStats, SpeedPerturb, AddNoise, AddReverb, CodecAugment, string, uuid, soundfile, csv
# were effectively removed when "Cell 4" was deleted.
# End of Cell 4


# Cell 5: Define Main Modal Training Function

logging.info("Defining Modal training function...")

@app.function(
    image=modal_image,
    gpu="A100-80GB",  # Change this in case you want to use a different GPU
    cpu=8,
    volumes={CHECKPOINT_DIR: volume},
    secrets=[hf_secret, wandb_secret],
    timeout=6 * 60 * 60,
    scaledown_window=30*60
)
def train_whisper_on_modal():
    global hparams 
    wandb_run = None 

    # === Robust W&B Initialization ===
    if hparams.get("use_wandb", False):
        logging.info("Attempting W&B Initialization...")
        try:
            import wandb
            from datetime import datetime
            
            # Log check for API key environment variable
            wandb_api_key = os.environ.get("WANDB_API_KEY")
            if wandb_api_key:
                logging.info("WANDB_API_KEY environment variable found.")
            else:
                 logging.warning("WANDB_API_KEY environment variable NOT found. W&B initialization might fail.")

            # Use standard string formatting to avoid f-string quote issues
            project_name = getattr(hparams, "wandb_project", "whisper-medium-egyptian-arabic") # Placeholder : Change this to your project's name on weights and biases
            entity_name = hparams.get("wandb_entity", None) # Use .get()
            resume_id = hparams.get("wandb_resume_id", None) # Get the resume ID

            # === Conditional W&B Init for resuming runs===
            if resume_id:
                logging.info(f"Attempting to resume W&B run with ID: {resume_id}")
                wandb_run = wandb.init(
                    project=project_name,
                    entity=entity_name,
                    id=resume_id,          # Pass the specific ID
                    resume="must",       # Ensure it resumes or fails
                    config=hparams,        # Still log config for reference
                    job_type="train",
                )
            else:
                logging.info("Starting a new W&B run.")
                logging.info("Calling wandb.init with project='{}', entity='{}'".format(project_name, entity_name))
                wandb_run = wandb.init(
                    project=project_name, # Use variable
                    entity=entity_name, # Use variable
                    config=hparams,
                    name=f"whisper-egy-{datetime.now().strftime('%Y%m%d-%H%M%S')}", # Generate name for new runs
                    job_type="train",
                    resume=False, # Explicitly false for new runs
                )
            # === End Conditional W&B Init ===
            
            # Log success or failure
            if wandb_run:
                 logging.info(f"WandB run initialized successfully: {wandb_run.name} ({wandb_run.id}) - View at {wandb_run.url}")
            else:
                 logging.error("wandb.init call completed but returned None. Initialization failed.")
                 hparams["use_wandb"] = False # Ensure W&B is marked as disabled

        except ImportError:
            logging.error("wandb library not found. Skipping WandB.", exc_info=True) # Add exc_info
            hparams["use_wandb"] = False
            wandb_run = None
        except Exception as e:
            logging.error(f"Could not initialize WandB due to an exception: {e}", exc_info=True)
            hparams["use_wandb"] = False
            wandb_run = None
    else:
        logging.info("Skipping W&B Initialization because hparams['use_wandb'] is False.")

    try: # Main training try block
        logging.info("=== Starting train_whisper_on_modal function ===")
        logging.info("Python version: %s", sys.version)
        logging.info("Current working directory: %s", os.getcwd())

        logging.info("Importing required libraries for training...")
        try:
            logging.info("Importing torch...")
            import torch
            logging.info(f"PyTorch version: {torch.__version__}")
            logging.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logging.info(f"CUDA device count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)} Properties: {torch.cuda.get_device_properties(i)}")
                logging.info(f"CUDA version: {torch.version.cuda}")

            logging.info("Importing speechbrain...")
            import speechbrain as sb # Retained for sb.utils.seed_everything
            logging.info(f"SpeechBrain version: {sb.__version__}")

            logging.info("Importing datasets...")
            # from datasets import load_dataset, DatasetDict, Audio, concatenate_datasets # Already imported globally
            logging.info("Datasets library imported successfully (globally).")

            logging.info("SpeechBrain data loading components (DynamicItemDataset, etc.) are no longer used.")


            logging.info("Importing utility libraries...")
            # import torch.optim as optim # Seq2SeqTrainer handles optimizer creation
            # import pandas as pd # No longer needed for add_duration
            import math # Retained - used for add_duration and potentially other math ops
            logging.info("Utility libraries (math) imported; optimizer handled by Trainer.")

            logging.info("Importing Unsloth and specific Transformers model class...")
            from unsloth import FastModel
            from transformers import WhisperForConditionalGeneration # Ensure this is imported
            logging.info("Unsloth and WhisperForConditionalGeneration imported successfully.")

            logging.info("All imports completed successfully")
        except Exception as e:
            logging.error(f"Error during imports: {e}", exc_info=True)
            raise

        # --- Setup: Folders, Seed ---
        try:
            logging.info("Accessing global variables and configurations...")
            output_folder = hparams.get("output_folder")
            save_folder = hparams.get("save_folder")
            logging.info(f"Output folder: {output_folder}")
            logging.info(f"Save folder: {save_folder}")

            if not output_folder or not save_folder:
                logging.error("output_folder or save_folder not defined in hparams.")
                return

            logging.info(f"Creating output directories: {output_folder}, {save_folder}")
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(save_folder, exist_ok=True)
            logging.info("Output directories created successfully")

            sb.utils.seed.seed_everything(hparams.get("seed", 1234))
            logging.info(f"Set random seed to {hparams.get('seed', 1234)}")
        except Exception as e:
            logging.error(f"Error in initial setup or seed setting: {e}", exc_info=True)
            raise

        # --- Load Main Dataset ---
        raw_datasets = None
        hf_dataset_id = hparams.get("hf_dataset_id")
        if not hf_dataset_id: logging.error("`hf_dataset_id` not found in hparams."); return
        try:
            logging.info(f"Loading HF dataset: {hf_dataset_id}")
            raw_datasets = load_dataset(hf_dataset_id, cache_dir=CACHE_DIR)
            logging.info(f"Raw datasets loaded:{raw_datasets}")
        except FileNotFoundError:
            logging.error(f"Dataset {hf_dataset_id} not found or cache dir issue."); return
        except Exception as e:
            logging.error(f"Error loading main dataset {hf_dataset_id}: {e}"); return

        # --- Add duration if missing ---
        def add_duration(batch):
            duration = 0.0
            try:
                audio_data = batch.get("audio")
                if audio_data and isinstance(audio_data, dict):
                    # 'array' is typically a NumPy array or list of floats/integers
                    # 'sampling_rate' is an int
                    array_data = audio_data.get("array")
                    sampling_rate = audio_data.get("sampling_rate")
                    if array_data is not None and sampling_rate is not None and sampling_rate > 0:
                        # len() works on NumPy arrays and lists
                        duration = len(array_data) / sampling_rate 
                batch["duration"] = duration
            except Exception as e: 
                logging.warning(f"Could not calculate duration for an item: {e}. Setting to {duration}. Audio data: {audio_data}", exc_info=True)
                batch["duration"] = duration # Ensure duration is always set
            return batch

        try:
            for split in raw_datasets:
                if 'duration' not in raw_datasets[split].column_names:
                    logging.info(f"Calculating duration for split: {split}")
                    try:
                        num_proc = min(os.cpu_count() if os.cpu_count() else 1, 4)
                        raw_datasets[split] = raw_datasets[split].map(add_duration, num_proc=num_proc)
                    except Exception as map_e:
                        logging.warning(f"Error calculating durations (multi-proc): {map_e}. Trying single process.")
                        raw_datasets[split] = raw_datasets[split].map(add_duration)
        except Exception as e: logging.error(f"Error processing dataset durations: {e}")

        # --- Instantiate Modules, Optimizer, Scheduler ---
        model = tokenizer = modules = optimizer = lr_scheduler = None # whisper_model changed to model, added tokenizer
        speechbrain_augmenter = None # For the extracted augmenter
        processed_datasets = None # For Hugging Face datasets
        data_collator = None # For Seq2SeqTrainer
        # compute_metrics will be defined locally after tokenizer is available
        try:
            logging.info("Initializing model with Unsloth, and SpeechBrain optimizer/scheduler...")
            
            # === Initialize Extracted Augmentations ===
            # This logic is moved here from WhisperFineTuneBrain.__init__
            # for DropChunk, DropFreq, DropBitResolution
            if hparams.get("augment", False):
                sb_augmentations_list_extracted = []
                initialized_augmentations_log_extracted = []

                if hparams.get("use_drop_chunk", False):
                    try:
                        drop_chunk_instance = DropChunk(
                            drop_length_low=hparams.get("drop_chunk_length_low", 1600),
                            drop_length_high=hparams.get("drop_chunk_length_high", 4800),
                            drop_count_low=hparams.get("drop_chunk_count_low", 1),
                            drop_count_high=hparams.get("drop_chunk_count_high", 5),
                        )
                        sb_augmentations_list_extracted.append(drop_chunk_instance)
                        initialized_augmentations_log_extracted.append("DropChunk")
                    except Exception as e:
                        logging.warning(f"Could not initialize DropChunk in train_whisper_on_modal: {e}. Skipping.")

                if hparams.get("use_drop_freq", False):
                    try:
                        drop_freq_instance = DropFreq(
                            drop_freq_count_low=hparams.get("drop_freq_count_low", 1),
                            drop_freq_count_high=hparams.get("drop_freq_count_high", 3),
                        )
                        sb_augmentations_list_extracted.append(drop_freq_instance)
                        initialized_augmentations_log_extracted.append("DropFreq")
                    except Exception as e:
                        logging.warning(f"Could not initialize DropFreq in train_whisper_on_modal: {e}. Skipping.")

                if hparams.get("use_drop_bit_resolution", False):
                    try:
                        bit_dropper_instance = DropBitResolution()
                        sb_augmentations_list_extracted.append(bit_dropper_instance)
                        initialized_augmentations_log_extracted.append("DropBitResolution")
                    except Exception as e:
                        logging.warning(f"Could not initialize DropBitResolution in train_whisper_on_modal: {e}. Skipping.")
                
                if sb_augmentations_list_extracted:
                    try:
                        effective_max_augmentations_extracted = min(
                            hparams.get("max_augmentations", len(sb_augmentations_list_extracted)),
                            len(sb_augmentations_list_extracted)
                        )
                        min_augmentations_extracted = hparams.get("min_augmentations", 1)
                        if effective_max_augmentations_extracted < min_augmentations_extracted:
                             effective_max_augmentations_extracted = min_augmentations_extracted
                        
                        if effective_max_augmentations_extracted > 0 : # only create if there are augmentations
                            speechbrain_augmenter = Augmenter(
                                parallel_augment=False,
                                concat_original=False,
                                min_augmentations=min_augmentations_extracted,
                                max_augmentations=effective_max_augmentations_extracted,
                                shuffle_augmentations=True,
                                augment_prob=hparams.get("augment_prob_master", 0.50),
                                augmentations=sb_augmentations_list_extracted,
                            )
                            logging.info(f"SpeechBrain Augmenter (extracted) initialized in train_whisper_on_modal with: {', '.join(initialized_augmentations_log_extracted)}")
                            logging.info(f"Applying min={min_augmentations_extracted}, max={effective_max_augmentations_extracted} from this pool per batch.")
                        else:
                            logging.info("No extracted SpeechBrain augmentations were initialized. Extracted Augmenter not created.")
                            speechbrain_augmenter = None

                    except Exception as e:
                        logging.warning(f"Could not initialize main Augmenter (extracted) in train_whisper_on_modal: {e}. Augmentation will be disabled for these.", exc_info=True)
                        speechbrain_augmenter = None
                else:
                    logging.info("No extracted SpeechBrain augmentations were enabled/initialized. Extracted Augmenter not created.")
            else:
                logging.info("Data augmentation disabled via hparams (augment=False). Extracted Augmenter not created.")

            # Continue with model loading...
            # Determine dtype
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            logging.info(f"Loading Whisper model with Unsloth using dtype: {dtype}")
            
            model, tokenizer = FastModel.from_pretrained(
                model_name=hparams.get("whisper_hub", "openai/whisper-large-v3"),  # Use hparam for model name
                dtype=dtype,
                load_in_4bit=False,  # IMPORTANT for full fine-tuning
                auto_model=WhisperForConditionalGeneration, # Specify the auto model class
                whisper_language=hparams.get("language", "ar"), # Get from hparams
                whisper_task=hparams.get("task", "transcribe"),   # Get from hparams
                # token = "hf_...", # Add if model is gated, but Whisper usually isn't
            )
            logging.info("Unsloth model and tokenizer loaded for full fine-tuning.")

            # Ensure the model's generation config is also updated
            # FastModel's from_pretrained for Whisper should handle setting language and task on the tokenizer,
            # which then configures the prefix tokens. Let's verify this and apply to model.generation_config.
            # The tokenizer returned by FastModel for Whisper is often a WhisperProcessor,
            # which contains the actual tokenizer as `tokenizer.tokenizer` or similar.
            # Or, it might be the tokenizer itself which has `set_prefix_tokens`.

            # Attempt to access the underlying tokenizer if it's a processor
            actual_tokenizer = tokenizer
            if hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'language'): # Common for Processor
                actual_tokenizer = tokenizer.tokenizer
            
            if hasattr(actual_tokenizer, 'language') and actual_tokenizer.language and \
               hasattr(actual_tokenizer, 'task') and actual_tokenizer.task:
                model.generation_config.language = actual_tokenizer.language
                model.generation_config.task = actual_tokenizer.task
                logging.info(f"Updated model.generation_config with language '{actual_tokenizer.language}' and task '{actual_tokenizer.task}' from Unsloth tokenizer.")
            else:
                 # Fallback: if language/task not found on tokenizer, set from hparams
                 model.generation_config.language = hparams.get("language", "ar")
                 model.generation_config.task = hparams.get("task", "transcribe")
                 logging.warning(f"Unsloth tokenizer structure unexpected or language/task not set. Set model.generation_config from hparams to lang='{hparams.get('language', 'ar')}', task='{hparams.get('task', 'transcribe')}'.")


            # The 'modules' dict is no longer needed as WhisperFineTuneBrain is removed.
            
            # Optimizer and LR Scheduler are now handled by Seq2SeqTrainer / TrainingArguments.
            # logging.info("Re-creating SpeechBrain-style optimizer for the new Unsloth model.") # Obsolete log
            # if model: # Ensure model is loaded
            #     no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
            #     optimizer_params = [
            #         {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            #          'weight_decay': hparams.get("weight_decay", 0.05), 'lr': hparams.get("learning_rate", 1e-7)},
            #         {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            #          'weight_decay': 0.0, 'lr': hparams.get("learning_rate", 1e-7)}
            #     ]
            #     optimizer = optim.AdamW(
            #         params=optimizer_params,
            #         lr=hparams.get("learning_rate", 1e-7),
            #         betas=(0.9, 0.999),
            #         eps=1e-8,
            #         weight_decay=hparams.get("weight_decay", 0.05)
            #     )
            #     logging.info("SpeechBrain-style AdamW optimizer re-created for the Unsloth model.")
            # else:
            #     logging.error("Model not loaded, cannot create optimizer.")
            #     optimizer = None # Ensure optimizer is None if model loading failed

            # lr_scheduler = sb.nnet.schedulers.NewBobScheduler(
            #     initial_value=hparams.get("learning_rate"), improvement_threshold=hparams.get("lr_improvement_threshold", 0.0025),
            #     annealing_factor=hparams.get("lr_annealing_factor"), patient=hparams.get("lr_patient", 0)
            # )
            logging.info("Model and tokenizer initialized. Optimizer and LR Scheduler handled by Trainer.")
        except KeyError as e: logging.error(f"Missing critical hparam for model/tokenizer setup: {e}"); return
        except Exception as e: logging.error(f"Error initializing model/tokenizer: {e}", exc_info=True); return

        # --- Checkpointer and Logger Setup --- (Old SB Checkpointer/Logger removed)
            
            # Instantiate Data Collator
            if tokenizer: # Ensure tokenizer (processor) is available
                data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=tokenizer)
                logging.info("DataCollatorSpeechSeq2SeqWithPadding instantiated.")
            else:
                logging.error("Tokenizer (WhisperProcessor) not available, cannot instantiate DataCollator.")
                # Potentially raise an error or handle this state if critical
                return # Cannot proceed without data_collator

            # Define compute_metrics function locally to capture tokenizer
            wer_metric_hf = evaluate.load("wer") 
            cer_metric_hf = evaluate.load("cer")

            def compute_metrics(pred):
                # pred.predictions are logit scores
                pred_ids = np.argmax(pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions, axis=-1)
                
                # pred.label_ids can be a tuple if pass_label_ids=True in TrainingArguments, handle that
                label_ids = pred.label_ids[0] if isinstance(pred.label_ids, tuple) else pred.label_ids
                
                # Replace -100 with the pad_token_id in label_ids
                label_ids[label_ids == -100] = tokenizer.pad_token_id

                # Decode predictions and labels
                pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
                label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)

                wer = 100 * wer_metric_hf.compute(predictions=pred_str, references=label_str)
                cer = 100 * cer_metric_hf.compute(predictions=pred_str, references=label_str)
                
                return {"wer": wer, "cer": cer}
            logging.info("compute_metrics function defined locally, WER/CER metrics loaded.")

            # Initialize num_map_workers from hparams, defaulting to os.cpu_count()
            num_map_workers = hparams.get("num_map_workers")
            if num_map_workers is None:
                num_map_workers = os.cpu_count() if os.cpu_count() is not None else 1
                # hparams["num_map_workers"] = num_map_workers # Not strictly necessary to save back if only used here
            logging.info(f"Using {num_map_workers} workers for dataset mapping.")

        # try: # Old SpeechBrain checkpointer/logger setup - REMOVED
            # epoch_counter = sb.utils.epoch_loop.EpochCounter(limit=hparams.get("epochs"))
            # checkpointer = sb.utils.checkpoints.Checkpointer(
            #     checkpoints_dir=save_folder,
            #     recoverables={ "model": modules["whisper"], "scheduler": lr_scheduler, "counter": epoch_counter, "optimizer": optimizer, },
            # )
            # train_logger = FileTrainLogger(save_file=os.path.join(output_folder, "train_log.txt"))
            # hparams["train_logger"] = train_logger 
            # logging.info("Initialized FileTrainLogger.")
        # except KeyError as e: 
        #     logging.error(f"Missing critical hparam for checkpointer/logger setup: {e}")
        
        # --- Load and Preprocess Hugging Face Datasets ---
        logging.info("Starting Hugging Face dataset loading and preprocessing...")
        
        hf_dataset_id_val = hparams.get("hf_dataset_id")
        if not hf_dataset_id_val:
            logging.error("`hf_dataset_id` not found in hparams. Exiting.")
            return 
        
        logging.info(f"Loading HF dataset: {hf_dataset_id_val} from cache_dir: {CACHE_DIR}")
        # raw_datasets was already loaded earlier, using that variable.
        # raw_hf_datasets = load_dataset(hf_dataset_id_val, cache_dir=CACHE_DIR) # This line re-loads, use existing raw_datasets
        raw_hf_datasets = raw_datasets # Use the already loaded raw_datasets
        logging.info(f"Raw Hugging Face datasets loaded: {raw_hf_datasets}")

        target_sr_val = hparams.get("target_sample_rate", 16000)
        for split_key_name in ["train_split", "valid_split", "test_split"]:
            split_name_val = hparams.get(split_key_name)
            if split_name_val and split_name_val in raw_hf_datasets:
                current_features = raw_hf_datasets[split_name_val].features
                if current_features.get('audio') and \
                   (not isinstance(current_features['audio'], Audio) or \
                    current_features['audio'].sampling_rate != target_sr_val):
                    logging.info(f"Casting and resampling audio column for split: {split_name_val} to {target_sr_val} Hz")
                    raw_hf_datasets[split_name_val] = raw_hf_datasets[split_name_val].cast_column(
                        "audio", Audio(sampling_rate=target_sr_val)
                    )
        
        text_column_name_val = hparams.get("text_column_name", "text")
        train_split_for_check = hparams.get("train_split", "train")
        if train_split_for_check in raw_hf_datasets and text_column_name_val not in raw_hf_datasets[train_split_for_check].column_names:
            logging.error(f"Text column '{text_column_name_val}' not found in dataset. Please check 'text_column_name' in hparams.")
            return

        def preprocess_function(examples):
            audio_inputs = examples["audio"]
            input_waveforms = [item["array"] for item in audio_inputs]
            # Assuming all items in batch have same sampling_rate after casting
            sampling_rate = audio_inputs[0]["sampling_rate"] if audio_inputs else target_sr_val

            processed_waveforms = []
            if speechbrain_augmenter is not None and hparams.get("augment", False):
                for waveform_array in input_waveforms:
                    waveform_tensor = torch.tensor(waveform_array, dtype=torch.float32).unsqueeze(0)
                    lengths = torch.tensor([1.0], device=waveform_tensor.device) # Relative length for SB
                    try:
                        augmented_wav, _ = speechbrain_augmenter(waveform_tensor, lengths)
                        processed_waveforms.append(augmented_wav.squeeze(0).cpu().numpy())
                    except Exception as aug_e_map:
                        logging.warning(f"SpeechBrain augmentation failed for an item in map: {aug_e_map}. Using original.")
                        processed_waveforms.append(waveform_array)
            else:
                processed_waveforms = input_waveforms
            
            # Unsloth's feature extractor (part of the tokenizer object)
            # The tokenizer here is the Unsloth-loaded one, which should be a WhisperProcessor.
            features = tokenizer.feature_extractor(
                processed_waveforms, sampling_rate=sampling_rate, return_tensors="pt", padding="longest"
            )
            
            # Unsloth's text tokenizer (part of the tokenizer object)
            # Accessing tokenizer.tokenizer for the actual WhisperTokenizerFast if tokenizer is a Processor
            actual_text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
            labels = actual_text_tokenizer(
                examples[text_column_name_val],
                padding="longest",
                truncation=True,
                return_tensors="pt"
            ).input_ids

            return {
                "input_features": features.input_features,
                "labels": labels,
            }

        processed_datasets = {}
        map_batch_size = hparams.get("preprocessing_batch_size", 16)
        
        train_split_name_map = hparams.get("train_split", "train")
        valid_split_name_map = hparams.get("valid_split", "validation")
        test_split_name_map = hparams.get("test_split", "test")
        
        columns_to_remove_map = None 
        if train_split_name_map in raw_hf_datasets:
            columns_to_remove_map = raw_hf_datasets[train_split_name_map].column_names
            logging.info(f"Processing train split: {train_split_name_map}. Removing columns: {columns_to_remove_map}")
            processed_datasets["train"] = raw_hf_datasets[train_split_name_map].map(
                preprocess_function, batched=True, batch_size=map_batch_size,
                num_proc=num_map_workers, remove_columns=columns_to_remove_map
            )
            logging.info(f"Train split processed. New columns: {processed_datasets['train'].column_names}")

        if valid_split_name_map in raw_hf_datasets:
            if columns_to_remove_map is None and valid_split_name_map in raw_hf_datasets : 
                 columns_to_remove_map = raw_hf_datasets[valid_split_name_map].column_names
            logging.info(f"Processing validation split: {valid_split_name_map}. Removing columns: {columns_to_remove_map}")
            processed_datasets["validation"] = raw_hf_datasets[valid_split_name_map].map(
                preprocess_function, batched=True, batch_size=map_batch_size,
                num_proc=num_map_workers, remove_columns=columns_to_remove_map
            )
            logging.info(f"Validation split processed. New columns: {processed_datasets['validation'].column_names}")

        if test_split_name_map and test_split_name_map in raw_hf_datasets:
            if columns_to_remove_map is None and test_split_name_map in raw_hf_datasets: 
                 columns_to_remove_map = raw_hf_datasets[test_split_name_map].column_names
            logging.info(f"Processing test split: {test_split_name_map}. Removing columns: {columns_to_remove_map}")
            processed_datasets["test"] = raw_hf_datasets[test_split_name_map].map(
                preprocess_function, batched=True, batch_size=map_batch_size,
                num_proc=num_map_workers, remove_columns=columns_to_remove_map
            )
            logging.info(f"Test split processed. New columns: {processed_datasets['test'].column_names}")
        
        logging.info("Hugging Face dataset loading and preprocessing finished.")

        # --- Comment out old SpeechBrain Dataloader/Dataset creation ---
        # logging.info("Creating SpeechBrain DynamicItemDatasets...")
        # datasets_dict = {}
        # output_keys = ["id", "signal_raw", "text_raw", "duration"]
        # try:
        #     required_splits = [hparams.get("train_split"), hparams.get("valid_split"), hparams.get("test_split")]
        #     required_splits = [s for s in required_splits if s]
        #     if not required_splits: logging.error("No dataset split names defined in hparams."); return
        #     for split in required_splits:
        #         if split in raw_datasets: # raw_datasets here is the original HF load
        #             dynamic_items = [audio_pipeline_minimal, text_pipeline_minimal]
        #             hf_dataset_split = raw_datasets[split]
        #             data_dict = {str(i): hf_dataset_split[i] for i in range(len(hf_dataset_split))}
        #             datasets_dict[split] = DynamicItemDataset(
        #                  data=data_dict, dynamic_items=dynamic_items, output_keys=output_keys,
        #             )
        #             logging.info(f"Successfully created DynamicItemDataset for split: {split} with {len(datasets_dict[split])} items.")
        #         else:
        #              logging.warning(f"Split '{split}' not found in loaded dataset. Skipping.")
        #     if not hparams.get("train_split") in datasets_dict or not hparams.get("valid_split") in datasets_dict:
        #          logging.error("Essential train or validation dataset could not be created. Exiting.")
        #          return
        # except Exception as e:
        #      logging.error(f"Error creating DynamicItemDatasets: {e}", exc_info=True)
        #      return

        # logging.info("Creating Dataloaders with DynamicBatchSampler...")
        # train_loader_kwargs = valid_loader_kwargs = test_loader_kwargs = None
        # try:
        #     dynamic_batching = hparams.get("batch_size_dynamic", True)
        #     loader_common_kwargs = {
        #         "num_workers": hparams.get("num_workers", 0),
        #         "pin_memory": True if hparams.get("num_workers", 0) > 0 else False,
        #         "prefetch_factor": 2 if hparams.get("num_workers", 0) > 0 else None,
        #         "collate_fn": PaddedBatch,
        #     }
        #     if dynamic_batching:
        #         max_batch_length_samples = int(hparams.get("max_batch_len_seconds") * hparams.get("target_sample_rate"))
        #         num_buckets = hparams.get("dynamic_batch_num_buckets", 60)
        #         logging.info(f"Using dynamic batching with max_batch_length={max_batch_length_samples} samples and num_buckets={num_buckets}.")

        #         def length_func(item_dict):
        #             duration = item_dict.get("duration")
        #             if duration is None or not isinstance(duration, (int, float)) or duration < 0: return 0
        #             return math.ceil(duration * hparams.get("target_sample_rate"))
                
        #         train_split_name_dl = hparams.get("train_split")
        #         if train_split_name_dl and train_split_name_dl in datasets_dict:
        #             train_sampler = DynamicBatchSampler(
        #                 datasets_dict[train_split_name_dl], max_batch_length=max_batch_length_samples,
        #                 num_buckets=num_buckets, shuffle=True, batch_ordering="random", length_func=length_func
        #             )
        #             train_loader_kwargs = { **loader_common_kwargs, "batch_sampler": train_sampler, "shuffle": False }
                
        #         valid_split_name_dl = hparams.get("valid_split")
        #         if valid_split_name_dl and valid_split_name_dl in datasets_dict:
        #             valid_sampler = DynamicBatchSampler(
        #                 datasets_dict[valid_split_name_dl], max_batch_length=max_batch_length_samples,
        #                 num_buckets=num_buckets, shuffle=False, batch_ordering="random", length_func=length_func
        #             )
        #             valid_loader_kwargs = { **loader_common_kwargs, "batch_sampler": valid_sampler, "shuffle": False }

        #         test_split_name_dl = hparams.get("test_split")
        #         if test_split_name_dl and test_split_name_dl in datasets_dict:
        #             test_sampler = DynamicBatchSampler(
        #                 datasets_dict[test_split_name_dl], max_batch_length=max_batch_length_samples,
        #                 num_buckets=num_buckets, shuffle=False, batch_ordering="random", length_func=length_func
        #             )
        #             test_loader_kwargs = { **loader_common_kwargs, "batch_sampler": test_sampler, "shuffle": False }
        #     else: # Static batching
        #         static_bs = hparams.get("loader_batch_size", 8)
        #         loader_common_kwargs["batch_size"] = static_bs
        #         if hparams.get("train_split") in datasets_dict: train_loader_kwargs = { **loader_common_kwargs, "shuffle": True }
        #         if hparams.get("valid_split") in datasets_dict: valid_loader_kwargs = { **loader_common_kwargs, "shuffle": False }
        #         if hparams.get("test_split") in datasets_dict: test_loader_kwargs = { **loader_common_kwargs, "shuffle": False }
            
        #     if not train_loader_kwargs or not valid_loader_kwargs:
        #         logging.error("Essential train or validation loader could not be created (SpeechBrain). Exiting.")
        #         return
        # except KeyError as e: logging.error(f"Missing critical hparam for SB dataloader setup: {e}"); return
        # except Exception as e: logging.error(f"Error creating SB Dataloaders/Samplers: {e}", exc_info=True); return

        # --- Initialize Brain --- (REMOVED)
        # whisper_brain = None
        # try:
            # logging.info("Initializing WhisperFineTuneBrain...")
            # ... (all Brain initialization code removed) ...
        # except Exception as e: logging.error(f"Error initializing WhisperFineTuneBrain: {e}", exc_info=True); return

        # --- WandB Watch Model (If enabled and wandb initialized) --- (REMOVED)
        # Check if wandb_run exists (meaning wandb.init was successful)
        # if wandb_run and hparams.get("wandb_watch_model", False):
            # ... (wandb.watch code removed) ...
        
        logging.info("Setting up Seq2SeqTrainingArguments...")
    
        # Determine floating point precision
        bf16_ready = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        # Get dataset splits for trainer
        train_dataset_for_trainer = processed_datasets.get(hparams.get("train_split", "train"))
        eval_dataset_for_trainer = processed_datasets.get(hparams.get("valid_split", "validation"))
        test_dataset_for_trainer = processed_datasets.get(hparams.get("test_split", "test"))

        if not train_dataset_for_trainer:
            logging.error("Train dataset not found in processed_datasets. Cannot start training.")
            return
        if not eval_dataset_for_trainer:
            logging.warning("Validation dataset not found in processed_datasets. Evaluation during training will be skipped.")

        training_args = Seq2SeqTrainingArguments(
            output_dir=hparams.get("output_folder", f"{CHECKPOINT_DIR}/whisper_large_egy_output_trainer"),
            per_device_train_batch_size=hparams.get("loader_batch_size", 4), 
            per_device_eval_batch_size=hparams.get("loader_batch_size", 4) * 2, 
            gradient_accumulation_steps=hparams.get("grad_accumulation_factor", 2),
            warmup_steps=hparams.get("lr_warmup_steps", 500),
            num_train_epochs=hparams.get("epochs", 3), 
            learning_rate=hparams.get("learning_rate", 1e-5),
            weight_decay=hparams.get("weight_decay", 0.01),
            
            fp16=not bf16_ready and torch.cuda.is_available(),
            bf16=bf16_ready and torch.cuda.is_available(),
            
            logging_steps=hparams.get("wandb_log_batch_freq", 25), 
            evaluation_strategy="steps" if eval_dataset_for_trainer else "no",
            eval_steps=hparams.get("eval_steps", 500) if eval_dataset_for_trainer else None, 
            save_strategy="steps",
            save_steps=hparams.get("save_steps", 500), 
            save_total_limit=hparams.get("num_checkpoints_to_keep", 2),
            
            optim=hparams.get("optimizer_type", "adamw_torch"), 
            
            remove_unused_columns=False, 
            label_names=['labels'], 
            
            report_to="wandb" if hparams.get("use_wandb", False) else "none",
            
            seed=hparams.get("seed", 1986),
            # generation_max_length = hparams.get("generation_max_length", 225), # Example, if needed for predict_with_generate
            # predict_with_generate=True, # To get generated output for WER/CER
        )

        logging.info("Initializing Seq2SeqTrainer...")
        trainer = Seq2SeqTrainer(
            model=model, 
            tokenizer=tokenizer.feature_extractor, 
            args=training_args,
            train_dataset=train_dataset_for_trainer,
            eval_dataset=eval_dataset_for_trainer,
            data_collator=data_collator, 
            compute_metrics=compute_metrics, 
        )

        logging.info("Starting training with Seq2SeqTrainer...")
        if train_dataset_for_trainer:
            try:
                trainer.train()
                logging.info("Training complete.")
                # Save final model
                final_save_path = os.path.join(training_args.output_dir, "final_model_checkpoint")
                trainer.save_model(final_save_path)
                logging.info(f"Final model saved to {final_save_path}")
                # Also save tokenizer
                tokenizer.save_pretrained(final_save_path)
                logging.info(f"Tokenizer saved to {final_save_path}")

                try:
                    volume.commit()
                    logging.info("Volume committed after training.")
                except Exception as commit_e:
                    logging.error(f"Error committing volume after training: {commit_e}")

            except Exception as fit_e:
                logging.error(f"Error during Seq2SeqTrainer training: {fit_e}", exc_info=True)
                try:
                    volume.commit()
                    logging.warning("Volume committed after training failure.")
                except Exception as commit_e:
                    logging.error(f"Error committing volume after training failure: {commit_e}")
        else:
            logging.warning("Training skipped as no train dataset was provided to the trainer.")

        if test_dataset_for_trainer:
            logging.info("Starting final evaluation on the test set...")
            try:
                eval_results = trainer.evaluate(test_dataset_for_trainer)
                logging.info(f"Evaluation results on test set: {eval_results}")
                # Log results to a file or W&B if needed
                results_path = os.path.join(training_args.output_dir, "test_results.json")
                import json
                with open(results_path, "w") as f:
                    json.dump(eval_results, f, indent=4)
                logging.info(f"Test results saved to {results_path}")

                try:
                    volume.commit()
                    logging.info("Volume committed after evaluation.")
                except Exception as commit_e:
                    logging.error(f"Error committing volume after evaluation: {commit_e}")
            except Exception as eval_e:
                logging.error(f"Error during final evaluation: {eval_e}", exc_info=True)
                try:
                    volume.commit()
                    logging.warning("Volume committed after evaluation failure.")
                except Exception as commit_e:
                    logging.error(f"Error committing volume after evaluation failure: {commit_e}")
        else:
            logging.info("Test set evaluation skipped as no test dataset was provided.")

        print("--- Modal Training Function Finished ---")
                except Exception as commit_e: logging.error(f"Error committing volume after training: {commit_e}")
            except Exception as fit_e:
                 logging.error(f"Error during training loop (fit): {fit_e}", exc_info=True)
                 try: volume.commit(); logging.warning("Volume committed after training failure.")
                 except Exception as commit_e: logging.error(f"Error committing volume after training failure: {commit_e}")
        else:
            missing = [item for item, flag in [("train dataset", bool(train_set)), ("validation dataset", bool(valid_set)),
                       ("train loader", bool(train_loader_kwargs)), ("validation loader", bool(valid_loader_kwargs)),
                       ("epoch counter", bool(epoch_counter)), ("brain instance", bool(whisper_brain))] if not flag]
            logging.warning(f"Training skipped: Missing components: {', '.join(missing)}.")

        # --- Evaluation ---
        test_set = datasets_dict.get(hparams.get("test_split"))
        if test_set and test_loader_kwargs and whisper_brain:
            logging.info("Starting final evaluation on test set...")
            try:
                if checkpointer:
                    try: checkpointer.recover_if_possible(min_key="WER"); logging.info("Loaded best checkpoint based on validation WER.")
                    except FileNotFoundError: logging.warning("No checkpoint found with min_key 'WER'. Evaluating current model state.")
                    except Exception as load_e: logging.warning(f"Could not load best checkpoint: {load_e}. Evaluating current model state.")
                else: logging.warning("Checkpointer not available. Evaluating current model state.")
                whisper_brain.evaluate(test_set=test_set, min_key="WER", test_loader_kwargs=test_loader_kwargs)
                logging.info("Evaluation complete.")
                try: volume.commit(); logging.info("Volume committed after evaluation.")
                except Exception as commit_e: logging.error(f"Error committing volume after evaluation: {commit_e}")
            except Exception as eval_e:
                 logging.error(f"Error during evaluation: {eval_e}", exc_info=True)
                 try: volume.commit(); logging.warning("Volume committed after evaluation failure.")
                 except Exception as commit_e: logging.error(f"Error committing volume after evaluation failure: {commit_e}")
        else:
            missing = [item for item, flag in [("test dataset", bool(test_set)),
                       ("test loader", bool(test_loader_kwargs)), ("brain instance", bool(whisper_brain))] if not flag]
            logging.warning(f"Evaluation skipped: Missing components: {', '.join(missing)}.")

        print("--- Modal Training Function Finished ---")

    except Exception as main_e:
        logging.error(f"Main training function failed: {main_e}", exc_info=True)
    finally:
        # --- Robust W&B Finish --- 
        if wandb_run: 
            try:
                logging.info(f"Finishing W&B run: {wandb_run.name}")
                wandb.finish()
            except Exception as wandb_finish_e:
                logging.error(f"Error during wandb.finish(): {wandb_finish_e}")
        else:
            logging.info("W&B run was not active, skipping wandb.finish().")

        # --- Commit Volume --- 
        try:
            volume.commit()
            logging.info("Final volume commit attempt in finally block.")
        except Exception as commit_e:
            logging.error(f"Error committing volume in finally block: {commit_e}")

# End of Cell 5

# Cell 6: Define Local Entrypoint (Unchanged)

@app.local_entrypoint()
def main():
    print("Submitting Whisper fine-tuning job to Modal...")
    try:
        train_whisper_on_modal.remote()
        print("Modal job submitted. Check Modal logs for progress.")
    except Exception as e:
         print(f"Error submitting job to Modal: {e}")

print("Modal local entrypoint defined.")
# End of Cell 6