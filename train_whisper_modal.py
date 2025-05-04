import logging
import datetime
import yaml  # Add yaml import
import tempfile  # Add tempfile for safe temporary file handling
import csv  # Add csv import for augmentation file handling
import shutil  # For cleanup
import math
import torch.nn.functional as F
from speechbrain.processing.signal_processing import  reverberate
from datasets import load_dataset  # Global import for dataset loading in Brain

# Turn off tokenizers parallelism warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Cell 1: Basic Modal Setup

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
def verify_whisper_tokens(model_name="openai/whisper-small", language="ar", task="transcribe"):
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
app = modal.App("name-of-your-app") # Placeholder : name your modal app here 

# --- Secrets ---
hf_secret = modal.Secret.from_name("huggingface-secret-write") # Define the huggingface secret object
wandb_secret = modal.Secret.from_name("wandb-secret") # Define the wandb secret object

# --- Persistent Storage ---
volume = modal.Volume.from_name(
    "name-of-your-volume", create_if_missing=True # Placeholder : name your volume here 
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
        "pip==23.3.2",
        "setuptools==69.0.3",
        "wheel==0.42.0",
        "pyarrow==15.0.0",      
        # Core ML libraries
        "torch==2.1.2",  # Latest stable that's well-tested with Whisper
        "torchaudio==2.1.2",  # Matching torch version
        "torchvision==0.16.2", # Add torchvision, compatible with torch 2.1.2
        "transformers==4.51.3",  # Use latest
        "accelerate==0.25.0",  # Latest stable
        "wandb",  
        # SpeechBrain and audio processing
        "speechbrain==1.0.3",  # Latest stable
        "librosa==0.10.1",  # Latest stable
        # Hugging Face ecosystem
        "datasets==2.16.1",  # Latest stable
        "huggingface_hub==0.30.0",  # Latest stable
        "sentencepiece==0.1.99",  # Latest stable
        # Additional dependencies
        "num2words==0.5.13",
        "pyyaml==6.0.1",
        "tqdm==4.66.1",
        "pandas==2.1.4",
        "soundfile==0.12.1"
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
token_info = verify_whisper_tokens(model_name="openai/whisper-small", language="ar", task="transcribe")

TARGET_SAMPLE_RATE = 16000 # Define target sample rate globally

hparams = {
    # Data
    "hf_dataset_id": "huggingfaceusername/datasetname", # Placeholder : Change this if you want to use a different dataset
    "train_split": "train",
    "valid_split": "validation",
    "test_split": "test",
    "target_sample_rate": TARGET_SAMPLE_RATE,

    # Model & Tokenizer
    "whisper_hub": "openai/whisper-small", # Placeholder : Change this if you want to use a different whisper model
    "save_folder": f"{CHECKPOINT_DIR}/whisper_small_egy_save",
    "output_folder": f"{CHECKPOINT_DIR}/whisper_small_egy_output",
    "language": "ar",
    "task": "transcribe",

    # Token IDs (Updated from verification using Processor)
    "sot_index": token_info["sot_token_id"] if token_info else 50258,
    "eos_index": token_info["eos_token_id"] if token_info else 50257,
    "pad_token_id": token_info["pad_token_id"] if token_info else 50257,
    "decoder_start_ids": token_info["decoder_start_ids"] if token_info else [50258, 50361, 50359, 50363],

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
    "loader_batch_size": 8, # Used only if batch_size_dynamic is False
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
    "use_wandb": True,  # Flag to enable/disable weights and biases
    "wandb_project": "you project's name on weights and biases ",  # Placeholder : Project name on weights and biases 
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

import torch
import torchaudio
import random
import speechbrain as sb
from speechbrain.utils.data_pipeline import takes, provides

print("Defining Data Loading Pipelines...")

@takes("audio")
@provides("signal_raw")
def audio_pipeline_minimal(audio):
    """Loads audio, resamples to target rate, and normalizes the signal."""
    try:
        if not audio or 'array' not in audio or 'sampling_rate' not in audio or audio['sampling_rate'] is None:
             logging.warning("Invalid or incomplete audio data in pipeline. Returning None.")
             return None
        sig = torch.tensor(audio["array"]).float()
        sr = audio["sampling_rate"]
        if sr != hparams["target_sample_rate"]:
            sig = torchaudio.functional.resample(sig, sr, hparams["target_sample_rate"])
        # Add check for empty signal after resampling
        if sig.numel() == 0:
             logging.warning("Signal became empty after resampling. Returning None.")
             return None
        # Normalize the signal to prevent large values
        if torch.max(torch.abs(sig)) > 0:
            sig = sig / torch.max(torch.abs(sig))
        else:
            logging.warning("Signal has zero max amplitude. Returning as is.")
        return sig
    except Exception as e:
         logging.error(f"Error processing audio in pipeline: {e}. Audio data: {audio}. Returning None.", exc_info=True)
         return None


@takes("text")
@provides("text_raw")
def text_pipeline_minimal(text):
    """Provides the raw text string."""
    return str(text) if text is not None else ""



print("Data Loading Pipelines defined.")
# End of Cell 3


# Cell 4: Define Brain Subclass 

import speechbrain as sb
from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.dataio.batch import PaddedBatch
# Consolidated imports
from speechbrain.augment.time_domain import (
    SpeedPerturb, AddNoise, AddReverb, DropChunk, DropFreq,
    DoClip, DropBitResolution
)
from speechbrain.augment.codec import CodecAugment 
from speechbrain.augment.augmenter import Augmenter
import os
import string
import torch
import torch.nn.utils.rnn as rnn_utils
import math
import uuid # Needed for noise file naming
import soundfile as sf # Needed for writing noise files
import csv # Needed for writing noise manifest
import torchaudio # Needed for resampling noise

# --- Wrapper Augmentation Modules ---

# --- Wrapper for Manual Pitch Shift ---
class PitchShiftWrapper(torch.nn.Module):
    def __init__(self, prob, pitch_steps_low, pitch_steps_high, sample_rate):
        super().__init__()
        self.prob = prob
        self.pitch_steps_low = pitch_steps_low
        self.pitch_steps_high = pitch_steps_high
        self.sample_rate = sample_rate

    def forward(self, waveforms, lengths=None): # Match Augmenter signature
        if random.random() < self.prob:
            n_steps = random.randint(self.pitch_steps_low, self.pitch_steps_high)
            if n_steps != 0:
                try:
                    # Apply pitch shift to the entire batch
                    # Ensure waveform tensor is on CPU if required by torchaudio
                    result = torchaudio.functional.pitch_shift(
                        waveforms.cpu(), self.sample_rate, n_steps=n_steps
                    ).to(waveforms.device)

                    if not torch.all(torch.isfinite(result)):
                        logging.warning("NaN/Inf detected after pitch shift. Using original waveforms.")
                        return waveforms
                    return result
                except Exception as e:
                    logging.warning(f"Batch torchaudio pitch shift failed: {e}. Using original waveforms.")
                    return waveforms
            else:
                # No shift applied if n_steps is 0
                return waveforms
        else:
            # Probability check failed, return original waveforms
            return waveforms

# --- Wrapper for Manual Gain ---
class GainWrapper(torch.nn.Module):
    def __init__(self, prob, gain_db_low, gain_db_high):
        super().__init__()
        self.prob = prob
        self.gain_db_low = gain_db_low
        self.gain_db_high = gain_db_high

    def forward(self, waveforms, lengths=None): # Match Augmenter signature
        if random.random() < self.prob:
            batch_size = waveforms.shape[0]
            # Generate a batch of random gain values in dB
            gain_db = torch.rand(batch_size, device=waveforms.device) * (self.gain_db_high - self.gain_db_low) + self.gain_db_low
            
            # Convert dB to amplitude factors
            # Avoid potential issues with very low dB values causing large negative exponents
            gain_amp = 10.0 ** (torch.clamp(gain_db, min=-80.0) / 20.0) # Clamp dB before exponentiation

            # Reshape gain_amp for broadcasting
            # Add dimensions to match waveform tensor (e.g., [batch, 1] or [batch, 1, 1])
            for _ in range(waveforms.dim() - 1):
                gain_amp = gain_amp.unsqueeze(-1)

            # Apply gain using vectorized multiplication
            result = waveforms * gain_amp

            # Apply clamp operation vectorized
            result = torch.clamp(result, min=-1.0, max=1.0)

            if not torch.all(torch.isfinite(result)):
                logging.warning("NaN/Inf detected after batched gain adjustment. Using original waveforms.")
                return waveforms
            return result
        else:
            # Probability check failed, return original waveforms
            return waveforms

class RIRSampler(torch.nn.Module):
    """Applies reverberation by sampling RIRs directly from an HF dataset object."""
    def __init__(self, rir_dataset, target_sample_rate=16000, rir_scale_factor=1.0):
        super().__init__()
        self.rir_dataset = rir_dataset
        self.num_rirs = len(rir_dataset) if rir_dataset is not None else 0
        self.target_sample_rate = target_sample_rate
        self.rir_scale_factor = rir_scale_factor
        self.max_len = int(3.0 * target_sample_rate)

    def forward(self, waveforms):
        if self.num_rirs == 0:
            return waveforms

        channel_added = False
        if waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(-1)
            channel_added = True

        idx = random.randint(0, self.num_rirs - 1)
        item = self.rir_dataset[idx]["audio"]
        rir = torch.tensor(item["array"], device=waveforms.device).float()
        sr = item["sampling_rate"]
        if sr != self.target_sample_rate:
            rir = torchaudio.functional.resample(rir, sr, self.target_sample_rate)

        if rir.dim() == 1:
            rir = rir.unsqueeze(-1)

        if rir.shape[0] > self.max_len:
            rir = rir[:self.max_len, :]

        if self.rir_scale_factor != 1:
            rir = F.interpolate(rir.transpose(0,1).unsqueeze(0), scale_factor=self.rir_scale_factor, mode="linear", align_corners=False).squeeze(0).transpose(0,1)

        rev = reverberate(waveforms, rir, rescale_amp="avg")
        return rev.squeeze(-1) if channel_added else rev


print("Defining WhisperFineTuneBrain...")

class WhisperFineTuneBrain(sb.Brain):
    def __init__(self, modules=None, opt_class=None, hparams=None, run_opts=None, checkpointer=None):
        # Ensure hparams is a dictionary or similar mapping (like SimpleNamespace)
        if not hasattr(hparams, '__dict__') and not isinstance(hparams, dict):
            raise TypeError(f"Expected hparams to be a dict or namespace-like, but got {type(hparams)}")

        super().__init__(modules, opt_class, hparams, run_opts, checkpointer)
        
        try:
            self.tokenizer = self.modules.whisper.tokenizer
            # Access hparams using getattr for safety after super init
            self.sot_index = getattr(self.hparams, "sot_index", 50258)  # Start of transcript token
            self.eos_index = getattr(self.hparams, "eos_index", 50257)  # End of text token
            self.pad_token_id = getattr(self.hparams, "pad_token_id", self.tokenizer.pad_token_id)
            self.decoder_start_ids = getattr(self.hparams, "decoder_start_ids", [self.sot_index])
            
            # Verify token IDs match tokenizer
            if self.tokenizer:
                # Convert special tokens to IDs for verification
                sot_id = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
                if self.sot_index != sot_id:
                    logging.warning(f"SOT index mismatch: hparams={self.sot_index}, tokenizer={sot_id}")
                if self.eos_index != self.tokenizer.eos_token_id:
                    logging.warning(f"EOS index mismatch: hparams={self.eos_index}, tokenizer={self.tokenizer.eos_token_id}")
                if self.pad_token_id != self.tokenizer.pad_token_id:
                    logging.warning(f"PAD token mismatch: hparams={self.pad_token_id}, tokenizer={self.tokenizer.pad_token_id}")
                
                # Verify decoder start sequence
                if len(self.decoder_start_ids) < 3:  # Should at least have [SOT, lang, task]
                    logging.warning(f"Decoder start sequence might be incomplete: {self.decoder_start_ids}")
                
        except AttributeError as e:
             logging.error(f"Missing critical hparam attribute for tokenizer/indices: {e}")
             raise ValueError(f"Missing critical hparam attribute for tokenizer/indices: {e}")
        except Exception as e:
             logging.error(f"Error initializing tokenizer/indices: {e}")
             raise

        # === Initialize SEPARATE metric objects ===
        self.wer_metric = ErrorRateStats()
        # Initialize CER metric specifically telling it to split tokens (chars)
        self.cer_metric = ErrorRateStats(split_tokens=True)

        self.augmenter = None
        initialized_augmentations = []

        # Create temporary directory for augmentation manifests (CSV files)
        temp_dir = tempfile.mkdtemp()
        self._temp_dir = temp_dir  # Store for cleanup

        # ***** Use getattr for hparams access *****
        if getattr(self.hparams, "augment", False):
            sb_augmentations = []  # List to hold augmenter instances

            target_sr = getattr(self.hparams, "target_sample_rate", TARGET_SAMPLE_RATE)

            # --- Dynamically Create Noise CSV and Initialize AddNoise (REVISED) ---
            noise_manifest_path = os.path.join(self._temp_dir, "noise_manifest.csv")
            if getattr(self.hparams, "use_add_noise", False):
                logging.info(f"Attempting AddNoise initialization. Dynamically creating noise manifest at: {noise_manifest_path}")
                fixed_noise_wav_dir = "/noise_assets" # Directory containing the downloaded WAVs
                try:
                    # List WAV files in the noise assets directory
                    noise_files = [f for f in os.listdir(fixed_noise_wav_dir) if f.endswith('.wav')]
                    if not noise_files:
                        logging.warning(f"No WAV files found in {fixed_noise_wav_dir} for noise augmentation. Skipping AddNoise.")
                        # Invalidate manifest path if no noise files are found
                        noise_manifest_path = None
                    else:
                        logging.info(f"Found noise files: {noise_files}")
                        # Write the noise manifest CSV
                        with open(noise_manifest_path, 'w', newline='', encoding='utf-8') as outfile:
                            writer = csv.writer(outfile)
                            # === CHANGE 1: Simplified header ===
                            header = ["ID", "duration", "wav", "wav_format", "wav_opts"]
                            writer.writerow(header)

                            # Add each noise file to the CSV
                            for noise_file in noise_files:
                                file_id = os.path.splitext(noise_file)[0] # Use filename as ID
                                full_path = os.path.join(fixed_noise_wav_dir, noise_file)
                                duration = 0.0
                                try:
                                    # Get audio info to determine duration
                                    info = torchaudio.info(full_path)
                                    duration = info.num_frames / info.sample_rate
                                    if duration <= 0:
                                        logging.warning(f"Calculated zero or negative duration for {noise_file}. Skipping.")
                                        continue # Skip files with invalid duration
                                except Exception as e_info:
                                    logging.warning(f"Could not get info/duration for {noise_file}: {e_info}. Skipping.", exc_info=True)
                                    continue # Skip files we can't get info for

                                # === CHANGE 2: Write the full_path ===
                                writer.writerow([file_id, f"{duration:.2f}", full_path, "wav", ""]) 
                            logging.info(f"Noise manifest created at {noise_manifest_path}.")

                        # Check if any rows were actually written (besides the header)
                        if os.path.exists(noise_manifest_path) and os.path.getsize(noise_manifest_path) > len(",".join(header).encode('utf-8')):
                            # === CHANGE 3: Remove replacements argument ===
                            noise_adder = AddNoise(
                                csv_file=noise_manifest_path, # Use the dynamically created manifest
                                snr_low=getattr(self.hparams, "noise_snr_low", 0), # Use getattr with default
                                snr_high=getattr(self.hparams, "noise_snr_high", 0), # Use getattr with default
                            )
                            sb_augmentations.append(noise_adder)
                            initialized_augmentations.append("AddNoise")
                            logging.info(f"Successfully initialized AddNoise with dynamically created manifest (using full paths).")
                        else:
                             logging.warning(f"Dynamically created noise manifest is empty or header-only. Skipping AddNoise initialization.")
                             noise_manifest_path = None # Invalidate the path


                except Exception as e_create_manifest:
                    logging.warning(f"Could not create noise manifest or initialize AddNoise: {e_create_manifest}. Skipping AddNoise.", exc_info=True)
                    noise_manifest_path = None # Ensure path is None on error

            # --- Initialize AddReverb using dynamically created manifest  ---
            reverb_prob = getattr(self.hparams, "reverb_prob", 0.0)
            rir_assets_dir = getattr(self.hparams, "rir_assets_dir", "/rir_assets") # Get dir from hparams
            rir_manifest_path = getattr(self.hparams, "rir_manifest_path", os.path.join(self._temp_dir, "rir_manifest.csv")) # Get manifest path

            if reverb_prob > 0:
                logging.info(f"Attempting to initialize AddReverb. Looking for RIR WAVs in: {rir_assets_dir}")
                try:
                    # Ensure the assets directory exists (should have been created in Dockerfile)
                    if not os.path.isdir(rir_assets_dir):
                         logging.warning(f"RIR assets directory {rir_assets_dir} not found. Skipping AddReverb.")
                    else:
                        rir_files = [f for f in os.listdir(rir_assets_dir) if f.endswith('.wav')]
                        if not rir_files:
                            logging.warning(f"No WAV files found in {rir_assets_dir}. Cannot create RIR manifest. Skipping AddReverb.")
                        else:
                            logging.info(f"Found RIR WAV files: {rir_files}")
                            # Create the RIR manifest CSV
                            with open(rir_manifest_path, 'w', newline='', encoding='utf-8') as outfile:
                                writer = csv.writer(outfile)
                                header = ["ID", "duration", "wav", "wav_format", "wav_opts"]
                                writer.writerow(header)

                                for rir_file in rir_files:
                                    file_id = os.path.splitext(rir_file)[0]
                                    full_path = os.path.join(rir_assets_dir, rir_file)
                                    duration = 0.0
                                    try:
                                        info = torchaudio.info(full_path)
                                        duration = info.num_frames / info.sample_rate
                                        if duration <= 0:
                                            logging.warning(f"Invalid duration for RIR {rir_file}. Skipping.")
                                            continue
                                    except Exception as e_info:
                                        logging.warning(f"Could not get info for RIR {rir_file}: {e_info}. Skipping.", exc_info=True)
                                        continue

                                    writer.writerow([file_id, f"{duration:.2f}", full_path, "wav", ""])
                            logging.info(f"RIR manifest created at {rir_manifest_path}.")

                            # Check if manifest is valid before initializing AddReverb
                            if os.path.exists(rir_manifest_path) and os.path.getsize(rir_manifest_path) > len(",".join(header).encode('utf-8')):
                                reverb_adder = AddReverb(
                                    csv_file=rir_manifest_path,
                                    rir_scale_factor=getattr(self.hparams, "rir_scale_factor", 1.0),
                                    reverb_sample_rate=target_sr, # Assume RIRs are at target SR
                                    clean_sample_rate=target_sr
                                )
                                sb_augmentations.append(reverb_adder)
                                initialized_augmentations.append("AddReverb")
                                logging.info("Successfully initialized AddReverb with dynamically created manifest.")
                            else:
                                logging.warning("Dynamically created RIR manifest is empty or invalid. Skipping AddReverb initialization.")

                except Exception as e_create_rir_manifest:
                    logging.warning(f"Could not create RIR manifest or initialize AddReverb: {e_create_rir_manifest}. Skipping AddReverb.", exc_info=True)


            # --- Initialize SpeedPerturb ---
            if getattr(self.hparams, "use_speed_perturb", False):
                try:
                    speeds = [factor / 100.0 for factor in self.hparams.speed_factors]
                    speed_perturber = SpeedPerturb(
                        orig_freq=target_sr,
                        speeds=speeds,
                        device=self.device 
                    )
                    sb_augmentations.append(speed_perturber)
                    initialized_augmentations.append("SpeedPerturb")
                except Exception as e:
                    logging.warning(f"Could not initialize SpeedPerturb: {e}. Skipping.", exc_info=True)

            # --- Initialize DropChunk ---
            if getattr(self.hparams, "use_drop_chunk", False):
                try:
                    drop_chunk = DropChunk(
                        drop_length_low=self.hparams.drop_chunk_length_low,
                        drop_length_high=self.hparams.drop_chunk_length_high,
                        drop_count_low=self.hparams.drop_chunk_count_low,
                        drop_count_high=self.hparams.drop_chunk_count_high,
                    )
                    sb_augmentations.append(drop_chunk)
                    initialized_augmentations.append("DropChunk")
                except Exception as e:
                    logging.warning(f"Could not initialize DropChunk: {e}. Skipping.", exc_info=True)

            # --- Initialize DropFreq ---
            if getattr(self.hparams, "use_drop_freq", False):
                try:
                    drop_freq = DropFreq(
                        drop_freq_count_low=self.hparams.drop_freq_count_low,
                        drop_freq_count_high=self.hparams.drop_freq_count_high,
                    )
                    sb_augmentations.append(drop_freq)
                    initialized_augmentations.append("DropFreq")
                except Exception as e:
                    logging.warning(f"Could not initialize DropFreq: {e}. Skipping.", exc_info=True)

            # --- Initialize PitchShiftWrapper ---
            if getattr(self.hparams, "use_pitch_shift", False):
                try:
                    pitch_shifter = PitchShiftWrapper(
                        prob=1.0, # Apply if selected by Augmenter
                        pitch_steps_low=self.hparams.pitch_steps_low,
                        pitch_steps_high=self.hparams.pitch_steps_high,
                        sample_rate=target_sr,
                    )
                    sb_augmentations.append(pitch_shifter)
                    initialized_augmentations.append("PitchShiftWrapper")
                except Exception as e:
                    logging.warning(f"Could not initialize PitchShiftWrapper: {e}. Skipping.", exc_info=True)

            # --- Initialize GainWrapper ---
            if getattr(self.hparams, "use_gain", False):
                try:
                    gain_adjuster = GainWrapper(
                        prob=1.0, # Apply if selected by Augmenter
                        gain_db_low=self.hparams.gain_db_low,
                        gain_db_high=self.hparams.gain_db_high,
                    )
                    sb_augmentations.append(gain_adjuster)
                    initialized_augmentations.append("GainWrapper")
                except Exception as e:
                    logging.warning(f"Could not initialize GainWrapper: {e}. Skipping.", exc_info=True)

            # --- Initialize DoClip  ---
            if getattr(self.hparams, "use_do_clip", False):
                try:
                    clipper = DoClip(
                        clip_low=self.hparams.clip_low,
                        clip_high=self.hparams.clip_high,
                    )
                    sb_augmentations.append(clipper)
                    initialized_augmentations.append("DoClip")
                except Exception as e:
                    logging.warning(f"Could not initialize DoClip: {e}. Skipping.", exc_info=True)

            # --- Initialize DropBitResolution  ---
            if getattr(self.hparams, "use_drop_bit_resolution", False):
                try:
                    bit_dropper = DropBitResolution()
                    sb_augmentations.append(bit_dropper)
                    initialized_augmentations.append("DropBitResolution")
                except Exception as e:
                    logging.warning(f"Could not initialize DropBitResolution: {e}. Skipping.", exc_info=True)

            # --- Initialize CodecAugment (Forcing g722)
            if getattr(self.hparams, "use_codec_augment", False):
                try:
                    codec_augmenter = CodecAugment(sample_rate=target_sr)
                    # Force g722 codec by overwriting the available list
                    codec_augmenter.available_format_encoders = [("g722", None)]
                    sb_augmentations.append(codec_augmenter)
                    initialized_augmentations.append("CodecAugment (g722 forced)")
                    logging.info(f"Initialized CodecAugment and forced g722 codec.")
                except Exception as e:
                    logging.warning(f"Could not initialize CodecAugment: {e}. Skipping.", exc_info=True)


            # --- Initialize Augmenter ---
            if sb_augmentations:
                try:
                    effective_max_augmentations = min(
                        getattr(self.hparams, "max_augmentations", len(sb_augmentations)),
                        len(sb_augmentations)
                    )
                    self.augmenter = Augmenter(
                        parallel_augment=False,
                        concat_original=False,
                        min_augmentations=getattr(self.hparams, "min_augmentations", 1),
                        max_augmentations=effective_max_augmentations,
                        shuffle_augmentations=True,
                        augment_prob=getattr(self.hparams, "augment_prob_master", 0.50), 
                        augmentations=sb_augmentations,
                    )
                    aug_names = [type(aug).__name__ for aug in sb_augmentations]
                    logging.info(f"SpeechBrain Augmenter initialized with {len(aug_names)} potential augmentations: {', '.join(aug_names)}")
                    logging.info(f"Applying min={getattr(self.hparams, 'min_augmentations', 1)}, max={effective_max_augmentations} from the pool per batch.")
                except Exception as e:
                    logging.warning(f"Could not initialize main Augmenter: {e}. Not initialized.", exc_info=True)
                    self.augmenter = None

        else:
            logging.info("Data augmentation disabled via hparams.")

    def __del__(self):
        """Clean up temporary files when the object is destroyed."""
        try:            
            if hasattr(self, '_temp_dir') and os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir)
                logging.info(f"Cleaned up temporary directory: {self._temp_dir}")
        except Exception as e:
            logging.warning(f"Error cleaning up temporary files: {e}")

    # --- _preprocess_batch (Apply Augmenter Batch-wise) ---
    def _preprocess_batch(self, batch, stage):
        """Applies augmentations and tokenization to the raw batch data."""
        try:
            batch = batch.to(self.device)
            if not isinstance(batch.signal_raw, (list, tuple)) or len(batch.signal_raw) != 2:
                 logging.error(f"Unexpected batch.signal_raw format: {type(batch.signal_raw)}")
                 raise ValueError("Invalid batch.signal_raw format")

            # Get original signals and lengths (relative)
            signals, signal_lens = batch.signal_raw
            ids = getattr(batch, 'id', [f'no_id_{i}' for i in range(signals.shape[0])])
            texts = batch.text_raw # Keep original texts for tokenization later
            if not isinstance(texts, list):
                 texts = [str(t) for t in texts] if hasattr(texts, '__iter__') else [str(texts)]

            wavs = signals  # Start with original signals
            wav_lens = signal_lens # Start with original relative lengths
            applied_augs_list = [] # List to store names of applied augmentations for logging
            augmentation_applied_this_batch = False # Flag to track if augmentation happened

            # ***** Apply SpeechBrain Augmenter to the whole batch *****
            if stage == sb.Stage.TRAIN and getattr(self.hparams, "augment", False) and self.augmenter is not None:
                # Check probability BEFORE calling augmenter
                if random.random() < self.augmenter.augment_prob:
                    try:
                        # --- Replicate selection logic for logging --- 
                        min_aug = max(0, self.augmenter.min_augmentations)
                        max_aug = max(min_aug, self.augmenter.max_augmentations)
                        num_available_augs = len(self.augmenter.augmentations)
                        
                        selected_augmentations_for_log = [] # Reset list for this batch
                        if max_aug > 0 and num_available_augs > 0:
                            N_augment = torch.randint(
                                low=min_aug,
                                high=min(max_aug, num_available_augs) + 1, # Ensure high doesn't exceed available
                                size=(1,),
                            ).item()

                            if N_augment > 0:
                                augmentations_lst = list(self.augmenter.augmentations.keys())
                                if self.augmenter.shuffle_augmentations:
                                    random.shuffle(augmentations_lst)
                                # Select the names
                                selected_augmentations_for_log = augmentations_lst[0:N_augment]
                                applied_augs_list = selected_augmentations_for_log # Store for logging after call
                                augmentation_applied_this_batch = True # Mark that we intend to augment
                                
                                # --- Now actually call the augmenter --- 
                                # NOTE: The augmenter performs its own selection internally.
                                # Our logging reflects the selection *predicted* by replicating the logic.
                                wavs, wav_lens = self.augmenter(wavs, wav_lens)

                        # Log the selection (or lack thereof) at DEBUG level
                        logging.debug(f"Augmenter Selection: Target N={N_augment if 'N_augment' in locals() else 'N/A'}, Selected Keys={selected_augmentations_for_log if selected_augmentations_for_log else 'None'}")

                    except Exception as aug_e:
                        logging.warning(f"Batch Augmentation failed: {aug_e}. Using original batch.", exc_info=True)
                        wavs = signals # Revert to original if augmentation fails
                        wav_lens = signal_lens
                        applied_augs_list = [f"Error: {type(aug_e).__name__}"]
                        augmentation_applied_this_batch = False # Augmentation failed

            # --- Log applied augmentations --- 
            # Log level can be adjusted (e.g., INFO more higher verbosity)
            logging.debug(f"Batch Augmentations Applied (Predicted): {applied_augs_list if augmentation_applied_this_batch else 'None'}")

            try:
                tokens_list = [self.tokenizer.encode(t) for t in texts]
                if any(not t for t in texts): logging.warning(f"Empty text found in batch: {texts}.")
                if any(not tk for tk in tokens_list): logging.warning(f"Empty token list after encoding: {texts} -> {tokens_list}.")

                bos_tokens_list = [torch.LongTensor(self.decoder_start_ids + t) for t in tokens_list]
                eos_tokens_list = [torch.LongTensor(t + [self.eos_index]) for t in tokens_list]
                target_tokens_list = [torch.LongTensor(t) for t in tokens_list]

            except Exception as e:
                 logging.error(f"Error encoding/creating token lists: {e}. Texts: {texts}")
                 raise ValueError(f"Error creating token lists: {e}")

            # Pad sequences
            try:
                tokens_bos_padded = rnn_utils.pad_sequence(bos_tokens_list, batch_first=True, padding_value=self.pad_token_id).to(self.device)
                tokens_eos_padded = rnn_utils.pad_sequence(eos_tokens_list, batch_first=True, padding_value=self.pad_token_id).to(self.device)
                tokens_padded = rnn_utils.pad_sequence(target_tokens_list, batch_first=True, padding_value=self.pad_token_id).to(self.device)
            except Exception as e:
                 logging.error(f"Error padding token sequences: {e}. BOS List lengths: {[len(t) for t in bos_tokens_list]}")
                 bsz = len(texts); max_len_fallback = 1
                 tokens_bos_padded = torch.full((bsz, max_len_fallback), self.pad_token_id, dtype=torch.long, device=self.device)
                 tokens_eos_padded = torch.full((bsz, max_len_fallback), self.pad_token_id, dtype=torch.long, device=self.device)
                 tokens_padded = torch.full((bsz, max_len_fallback), self.pad_token_id, dtype=torch.long, device=self.device)

            # **IMPORTANT**: Calculate token lengths based on the *original* token lists, NOT the potentially modified wav_lens
            bos_max_len = tokens_bos_padded.shape[1]
            eos_max_len = tokens_eos_padded.shape[1]
            tok_max_len = tokens_padded.shape[1]

            tokens_bos_lens_rel = (torch.tensor([len(t) for t in bos_tokens_list], device=self.device, dtype=torch.float) / bos_max_len
                                   if bos_max_len > 0 else torch.zeros(tokens_bos_padded.shape[0], device=self.device, dtype=torch.float))
            tokens_eos_lens_rel = (torch.tensor([len(t) for t in eos_tokens_list], device=self.device, dtype=torch.float) / eos_max_len
                                   if eos_max_len > 0 else torch.zeros(tokens_eos_padded.shape[0], device=self.device, dtype=torch.float))
            tokens_lens_rel = (torch.tensor([len(t) for t in target_tokens_list], device=self.device, dtype=torch.float) / tok_max_len
                               if tok_max_len > 0 else torch.zeros(tokens_padded.shape[0], device=self.device, dtype=torch.float))

            # Package into tuples
            tokens_bos = (tokens_bos_padded, tokens_bos_lens_rel)
            tokens_eos = (tokens_eos_padded, tokens_eos_lens_rel)
            tokens = (tokens_padded, tokens_lens_rel) # For metrics reference

            # Return the potentially augmented wavs and their corresponding lengths
            return wavs.to(self.device), wav_lens.to(self.device), tokens_bos, tokens_eos, tokens, texts

        except Exception as e:
            logging.error(f"Critical Error in _preprocess_batch: {e}", exc_info=True)
            # Attempt to return dummy data matching expected types
            bsz = batch.batch_size if hasattr(batch, 'batch_size') else 1
            dummy_wavs = torch.zeros((bsz, 1), device=self.device)
            dummy_lens = torch.zeros(bsz, device=self.device)
            dummy_tokens = torch.full((bsz, 1), self.pad_token_id, dtype=torch.long, device=self.device)
            dummy_tok_lens = torch.zeros(bsz, device=self.device, dtype=torch.float)
            dummy_texts = ["preprocessing_error"] * bsz
            return (dummy_wavs, dummy_lens,
                    (dummy_tokens, dummy_tok_lens), (dummy_tokens, dummy_tok_lens),
                    (dummy_tokens, dummy_tok_lens), dummy_texts)


    # --- compute_forward (Using getattr for hparams access) ---
    def compute_forward(self, batch, stage):
        """Runs preprocessing and the Whisper model forward pass."""
        try:
            # Call preprocess ONCE and get all required outputs
            wavs, wav_lens, tokens_bos_data, tokens_eos_data, tokens_data, texts = self._preprocess_batch(batch, stage)
            tokens_bos, _ = tokens_bos_data # Unpack for model input

            if wavs is None or tokens_bos is None or wavs.numel() == 0 or tokens_bos.numel() == 0:
                 logging.error("Empty/None tensors from preprocessing in compute_forward.")
                 bsz = batch.batch_size if hasattr(batch, 'batch_size') else 1
                 dummy_log_probs = torch.zeros((bsz, 1, self.tokenizer.vocab_size), device=self.device)
                 dummy_wav_lens = torch.zeros(bsz, device=self.device)
                 # Return dummy data matching the NEW expected structure
                 dummy_tokens = torch.full((bsz, 1), self.pad_token_id, dtype=torch.long, device=self.device)
                 dummy_tok_lens = torch.zeros(bsz, device=self.device, dtype=torch.float)
                 dummy_texts = ["preprocessing_error"] * bsz
                 return (dummy_log_probs, None, dummy_wav_lens,
                         (dummy_tokens, dummy_tok_lens), (dummy_tokens, dummy_tok_lens), dummy_texts)

            enc_out, logits, hyps = self.modules.whisper(wavs, tokens_bos)

            # Directly compute log_softmax
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Check after log_softmax for numerical stability
            if not torch.all(torch.isfinite(log_probs)):
                logging.error("NaN/Inf detected AFTER log_softmax. Loss will likely be NaN. Check model internal stability.")

            # --- Simple Greedy Decoding for evaluation ---
            hyps_out = None
            if stage != sb.Stage.TRAIN:
                try:
                    # Greedy selection of most probable token at each timestep
                    pred_tokens = torch.argmax(logits, dim=-1)  # [B, T]
                    hyps_out = pred_tokens.detach().cpu().tolist()  # Convert to list for tokenizer.decode
                except Exception as dec_e:
                    logging.error(f"Error during greedy decoding: {dec_e}")
                    hyps_out = None

            # Return model outputs AND the target info from the single preprocess call
            return log_probs, hyps_out, wav_lens.to(self.device), tokens_eos_data, tokens_data, texts

        except Exception as e:
             logging.error(f"Error in compute_forward: {e}", exc_info=True)
             bsz = batch.batch_size if hasattr(batch, 'batch_size') else 1
             dummy_log_probs = torch.zeros((bsz, 1, self.tokenizer.vocab_size), device=self.device)
             dummy_wav_lens = torch.zeros(bsz, device=self.device)
             # Return dummy data matching the NEW expected structure
             dummy_tokens = torch.full((bsz, 1), self.pad_token_id, dtype=torch.long, device=self.device)
             dummy_tok_lens = torch.zeros(bsz, device=self.device, dtype=torch.float)
             dummy_texts = ["preprocessing_error"] * bsz
             return (dummy_log_probs, None, dummy_wav_lens,
                     (dummy_tokens, dummy_tok_lens), (dummy_tokens, dummy_tok_lens), dummy_texts)


    # --- compute_objectives (Mostly unchanged, relies on preprocessing fix) ---
    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss and evaluation metrics."""
        try:
            # Unpack predictions from compute_forward (now includes targets)
            log_probs, hyps, wav_lens, tokens_eos_data, tokens_data, target_words_raw = predictions

            if log_probs is None or not torch.all(torch.isfinite(log_probs)):
                 logging.error("Invalid log_probs (None, NaN/Inf) in compute_objectives. Returning zero loss.")
                 return torch.tensor(0.0, device=self.device, requires_grad=False)

            ids = getattr(batch, 'id', [f'no_id_{i}' for i in range(log_probs.shape[0])])
            
            # Directly unpack target tokens from predictions
            if tokens_eos_data is None or tokens_eos_data[0] is None:
                 logging.error("Missing target token data (tokens_eos_data) in predictions.")
                 return torch.tensor(0.0, device=self.device, requires_grad=False)
            tokens_eos_padded, tokens_eos_lens_rel = tokens_eos_data
            # Unpack original tokens too, if needed for future metrics (not currently used in loss)
            if tokens_data is None or tokens_data[0] is None:
                 logging.warning("Missing original token data (tokens_data) in predictions.")
                 # Set defaults if needed, though not used for loss calculation
                 tokens_padded = None
                 tokens_lens_rel = None
            else:
                 tokens_padded, tokens_lens_rel = tokens_data

            if tokens_eos_padded is None or tokens_eos_padded.numel() == 0:
                logging.error("Empty target tensor (tokens_eos_padded) in compute_objectives. Returning zero loss.")
                return torch.tensor(0.0, device=self.device, requires_grad=False)

            # --- Calculate Loss ---
            loss = torch.tensor(0.0, device=self.device, requires_grad=stage==sb.Stage.TRAIN)
            try:
                use_lengths = (isinstance(tokens_eos_lens_rel, torch.Tensor) and
                               tokens_eos_lens_rel.numel() > 0 and
                               torch.all(tokens_eos_lens_rel > 0))

                if use_lengths:
                    logging.debug(f"Calling nll_loss with log_probs shape {log_probs.shape} and target shape {tokens_eos_padded.shape}, length tensor shape {tokens_eos_lens_rel.shape}")
                    loss = sb.nnet.losses.nll_loss(log_probs, tokens_eos_padded, length=tokens_eos_lens_rel)
                else:
                    logging.debug(f"Calling nll_loss (no length) with log_probs shape {log_probs.shape} and target shape {tokens_eos_padded.shape}")
                    logging.warning("Invalid target lengths for NLL loss. Using full padded length.")
                    loss = sb.nnet.losses.nll_loss(log_probs, tokens_eos_padded)

                if not torch.isfinite(loss):
                    logging.error("NaN or Inf loss calculated. Log Probs finite: {}, Targets: {}. Returning zero loss.".format(torch.all(torch.isfinite(log_probs)), tokens_eos_padded.shape))
                    loss = torch.tensor(0.0, device=self.device, requires_grad=stage==sb.Stage.TRAIN)

            except Exception as loss_e:
                logging.error(f"Error calculating NLL loss: {loss_e}")
                loss = torch.tensor(0.0, device=self.device, requires_grad=stage==sb.Stage.TRAIN)

            # --- Calculate Metrics (WER/CER) ---
            if stage != sb.Stage.TRAIN and hyps is not None:
                try:
                    predicted_words = self.tokenizer.batch_decode(hyps, skip_special_tokens=True)
                    target_words = target_words_raw # Raw text list

                    if not isinstance(predicted_words, list): predicted_words = [str(predicted_words)]
                    if not isinstance(target_words, list): target_words = [str(target_words)]
                    predicted_words = [str(p) for p in predicted_words]
                    target_words = [str(t) for t in target_words]

                    if len(ids) == len(predicted_words) == len(target_words):
                         self.wer_metric.append(ids, predicted_words, target_words)
                         self.cer_metric.append(ids, predicted_words, target_words)
                    else:
                         logging.warning(f"Mismatch lengths for WER/CER: ids({len(ids)}), preds({len(predicted_words)}), targets({len(target_words)}). Skipping.")

                except Exception as metric_e: logging.error(f"Error calculating WER/CER metrics: {metric_e}")

            if not isinstance(loss, torch.Tensor):
                logging.error(f"Loss is not a tensor ({type(loss)}). Returning zero.")
                loss = torch.tensor(0.0, device=self.device, requires_grad=stage==sb.Stage.TRAIN)

            return loss

        except Exception as e:
             logging.error(f"Error in compute_objectives: {e}", exc_info=True)
             return torch.tensor(0.0, device=self.device, requires_grad=stage==sb.Stage.TRAIN)


    # --- on_stage_start (Initialize BOTH metric objects) ---
    def on_stage_start(self, stage, epoch):
        try:
            # Re-initialize both at the start of validation/test stages
            self.wer_metric = ErrorRateStats()
            self.cer_metric = ErrorRateStats(split_tokens=True)
            logging.info(f"Initialized WER/CER metrics for stage {stage}.")
        except Exception as e: 
            logging.error(f"Error initializing metrics for stage {stage}: {e}")

    # --- on_stage_end  ---
    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {}
        try:
            # --- Get current LR ---
            current_lr = 'N/A'
            if hasattr(self, 'optimizer') and self.optimizer and hasattr(self.optimizer, 'param_groups') and self.optimizer.param_groups:
                try: 
                    current_lr = self.optimizer.param_groups[0]['lr']
                except Exception: 
                    logging.warning("Could not retrieve learning rate.")

            # === Store Train Loss when TRAIN stage ends ===
            if stage == sb.Stage.TRAIN:
                self._current_epoch_train_loss = getattr(self, 'avg_train_loss', 0.0)
                stage_stats["train_loss_epoch_avg"] = self._current_epoch_train_loss

            # === Log metrics when VALID or TEST stage ends ===
            elif stage == sb.Stage.VALID or stage == sb.Stage.TEST:
                stage_key_prefix = "valid" if stage == sb.Stage.VALID else "test"
                stage_stats["loss"] = stage_loss
                wer = cer = float('inf')  # Initialize to infinity
                try:
                    # Summarize from separate objects
                    wer = self.wer_metric.summarize("error_rate")  # Default is WER
                    cer = self.cer_metric.summarize("error_rate")  # Default is also error_rate, but calculated on chars

                    stage_stats["WER"] = wer if math.isfinite(wer) else float('inf')
                    stage_stats["CER"] = cer if math.isfinite(cer) else float('inf')
                except Exception as e:
                    logging.error(f"Error summarizing {stage_key_prefix} WER/CER metrics: {e}")
                    # Keep WER/CER as infinity

                # --- Log to File Logger (or WandbLogger via log_stats) ---
                if stage == sb.Stage.VALID:
                    try:
                        train_loss_to_log = getattr(self, '_current_epoch_train_loss', 0.0)
                        file_logger = getattr(self.hparams, "train_logger", None) # Will be FileTrainLogger or WandbLogger
                        if file_logger:
                            file_logger.log_stats(
                                stats_meta={"epoch": epoch, "lr": current_lr},
                                train_stats={'loss': train_loss_to_log}, # Pass avg train loss
                                valid_stats=stage_stats,  # Contains loss, WER, CER
                            )
                    except Exception as e:
                        logging.error(f"Error logging validation stats via logger: {e}")
                elif stage == sb.Stage.TEST:
                    file_logger = getattr(self.hparams, "train_logger", None) # Will be FileTrainLogger or WandbLogger
                    if file_logger:
                        loaded_epoch = getattr(self.hparams.epoch_counter, 'current', 'N/A')
                        file_logger.log_stats(
                            stats_meta={"Epoch loaded": loaded_epoch}, 
                            test_stats=stage_stats
                        )

                # --- Manual W&B logging for epoch stats ---
                if getattr(self.hparams, "use_wandb", False):
                    try:
                        import wandb
                        # Check if wandb.run is active (initialized successfully earlier)
                        if wandb.run: 
                            if stage == sb.Stage.VALID:
                                # Ensure train_loss_to_log is available
                                train_loss_to_log = getattr(self, '_current_epoch_train_loss', 0.0)
                                wandb_metrics = {
                                    "epoch": epoch,
                                    "learning_rate": current_lr if isinstance(current_lr, (int, float)) else -1.0,
                                    "loss/train_epoch": train_loss_to_log,
                                    f"loss/{stage_key_prefix}_epoch": stage_stats.get("loss", float('inf')),
                                    f"error_rate/{stage_key_prefix}_WER": stage_stats.get("WER", float('inf')),
                                    f"error_rate/{stage_key_prefix}_CER": stage_stats.get("CER", float('inf')),
                                }
                            elif stage == sb.Stage.TEST:
                                wandb_metrics = {
                                    f"final/{stage_key_prefix}_loss": stage_stats.get("loss", float('inf')),
                                    f"final/{stage_key_prefix}_WER": stage_stats.get("WER", float('inf')),
                                    f"final/{stage_key_prefix}_CER": stage_stats.get("CER", float('inf')),
                                }
                            else: # Should not happen in this elif block, but safety first
                                wandb_metrics = {}
                            
                            if wandb_metrics: # Only log if we have metrics
                                # Convert torch tensors to float for W&B compatibility
                                wandb_metrics_processed = {}
                                for k, v in wandb_metrics.items():
                                    if isinstance(v, torch.Tensor):
                                        try:
                                            v = v.item()
                                        except Exception:
                                            continue  # Skip if can't convert
                                    wandb_metrics_processed[k] = v
                                wandb_metrics_clean = {k: v for k, v in wandb_metrics_processed.items() if isinstance(v, (int, float)) and math.isfinite(v)}
                                if wandb_metrics_clean:
                                    wandb.log(wandb_metrics_clean)
                                    logging.info(f"Logged {stage_key_prefix} epoch metrics to WandB.")
                        else:
                            logging.warning("WandB run not active, skipping epoch log.")
                    except ImportError:
                        logging.warning("wandb library not found during epoch logging.")
                    except Exception as e:
                        logging.error(f"Error logging {stage_key_prefix} stats to WandB: {e}")

                # --- Checkpointing (on VALID stage) ---
                if stage == sb.Stage.VALID:
                    num_to_keep = getattr(self.hparams, "num_checkpoints_to_keep", 1)
                    if wer is not None and math.isfinite(wer) and hasattr(self, 'checkpointer') and self.checkpointer:
                        try:
                            self.checkpointer.save_and_keep_only(
                                meta={"WER": wer}, 
                                min_keys=["WER"], 
                                num_to_keep=num_to_keep
                            )
                        except Exception as e: 
                            logging.error(f"Error saving checkpoint: {e}")

        except Exception as e:
            is_wandb_enabled = getattr(self.hparams, "use_wandb", False)
            logging.error(f"Error in on_stage_end for stage {stage} (W&B enabled: {is_wandb_enabled}): {e}", exc_info=True)

        # Clear metrics after VALID or TEST stage finishes logging them
        if stage == sb.Stage.VALID or stage == sb.Stage.TEST:
            if hasattr(self, "wer_metric"): 
                self.wer_metric.clear()
            if hasattr(self, "cer_metric"): 
                self.cer_metric.clear()

        return stage_stats

    def on_fit_start(self):
        try:
            super().on_fit_start()
            self.avg_train_loss = 0.0 # Still needed for on_train_epoch_end
            self._train_loss_buffer = []
            self._current_epoch_train_loss = 0.0 # Initialize the new variable
            logging.info("Fit started, initialized train loss tracking.")
        except Exception as e: logging.error(f"Error in on_fit_start: {e}")


    # --- fit_batch (Extreme Gradient Stabilization) ---
    def fit_batch(self, batch):
        if not hasattr(self, 'optimizer') or not self.optimizer:
            logging.error("Optimizer not found in fit_batch!")
            return torch.tensor(0.0, device=self.device).cpu()

        # --- Boilerplate and Warmup ---
        grad_accum = getattr(self.hparams, "grad_accumulation_factor", 1)  # Default 1 if not set
        should_step = (self.step + 1) % grad_accum == 0  # Correct step check for grad accum
        loss = torch.tensor(0.0, device=self.device)
        
        # Learning rate warmup
        warmup_steps = getattr(self.hparams, "lr_warmup_steps", 0)  # Default 0
        if hasattr(self, 'step_counter'):
            self.step_counter += 1
        else:
            self.step_counter = 1
            
        # Apply learning rate warmup if we're in warmup phase
        if self.step_counter < warmup_steps and hasattr(self, 'optimizer'):
            warmup_factor = self.step_counter / max(1, warmup_steps)
            base_lr = getattr(self.hparams, "learning_rate", 1e-5)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = base_lr * warmup_factor

        try:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            batch_loss_value = loss.detach().item()  # Get value for logging

            # --- Loss Handling & Backward ---
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                if not torch.isfinite(loss):
                    logging.warning(f"Non-finite loss detected: {loss.item()}. Skipping backward/step.")
                    batch_loss_value = float('nan')  # Record NaN for logging
                    loss = torch.tensor(0.0, device=self.device)  # Avoid error in buffer append
                else:
                    try:
                        scaled_loss = loss / grad_accum
                        scaled_loss.backward()
                    except Exception as e:
                        logging.error(f"Error during backward(): {e}. Skipping step.")
                        if self.optimizer:
                            self.optimizer.zero_grad(set_to_none=True)
                        should_step = False
            else:
                should_step = False  # Don't step if loss wasn't valid for backward
                batch_loss_value = float('nan')  # Record NaN
                if not isinstance(loss, torch.Tensor):
                    loss = torch.tensor(0.0, device=self.device)

            # --- Grad Norm Calc, Clipping & Optimizer Step ---
            current_grad_norm = float('nan')  # Initialize for logging
            if should_step:  # Only if loss was valid and it's time to step
                try:
                    # Calculate gradient norm BEFORE clipping for logging
                    total_norm_sq = 0.0
                    max_norm_val = 0.0
                    max_norm_name = None
                    for n, p in self.modules.named_parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2).item()
                            total_norm_sq += param_norm ** 2
                            if param_norm > max_norm_val:
                                max_norm_val = param_norm
                                max_norm_name = n
                    current_grad_norm = math.sqrt(total_norm_sq)

                    # Clip Gradients
                    max_norm = getattr(self.hparams, "max_grad_norm", 5.0)
                    torch.nn.utils.clip_grad_norm_(self.modules.parameters(), max_norm)

                    # Optimizer Step
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                except Exception as e:
                    logging.error(f"Error during optimizer step/clipping/zero_grad: {e}")
                    try:
                        if self.optimizer:
                            self.optimizer.zero_grad(set_to_none=True)
                    except Exception as zg_e:
                        logging.error(f"Error during zero_grad after step error: {zg_e}")

            # --- Log Batch Metrics to W&B ---
            log_freq = getattr(self.hparams, "wandb_log_batch_freq", 0)  # Default 0 (disabled)
            if getattr(self.hparams, "use_wandb", False) and log_freq > 0 and (self.step + 1) % log_freq == 0:
                try:
                    import wandb
                    if wandb.run:
                        wandb_step_metrics = {
                            "train/batch_loss": batch_loss_value,
                            "train/gradient_norm": current_grad_norm,
                            "train/max_grad_norm": max_norm_val if 'max_norm_val' in locals() else float('nan'),
                            "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                            "trainer/global_step": self.step,
                            "trainer/should_step": int(should_step),
                        }         
                        # Only log finite numerical values
                        wandb.log({k: v for k, v in wandb_step_metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)})

                            
                except Exception as wandb_log_e:
                    logging.warning(f"Could not log step metrics to W&B: {wandb_log_e}")

            # --- Accumulate loss for epoch average ---
            self._train_loss_buffer.append(batch_loss_value)

        except Exception as e:
            logging.error(f"Error in fit_batch: {e}", exc_info=True)
            loss = torch.tensor(float('nan'))  # Return NaN on error

        return loss.cpu() if isinstance(loss, torch.Tensor) else torch.tensor(loss)


    def on_train_epoch_end(self, epoch):
        # Define a debug file path (use /tmp which is usually writable)
        debug_file_path = f"/tmp/epoch_{epoch}_debug.txt"
        try:
            # === START: Write to debug file ===
            with open(debug_file_path, "a") as f: # Open in append mode
                f.write(f"--- DEBUG FILE: Entering on_train_epoch_end for epoch {epoch} ---\n")
                buffer_len_debug = len(self._train_loss_buffer)
                f.write(f"DEBUG FILE: Current self._train_loss_buffer length = {buffer_len_debug}\n")

                if self._train_loss_buffer:
                    f.write(f"DEBUG FILE: Buffer content sample (first 5): {self._train_loss_buffer[:5]}\n")
                    f.write(f"DEBUG FILE: Buffer content sample (last 5): {self._train_loss_buffer[-5:]}\n")
                    try:
                        buffer_sum = sum(self._train_loss_buffer)
                        buffer_len = len(self._train_loss_buffer)
                        f.write(f"DEBUG FILE: Calculated buffer sum = {buffer_sum}\n")
                        f.write(f"DEBUG FILE: Calculated buffer len = {buffer_len}\n")
                        if buffer_len > 0:
                            calculated_avg = buffer_sum / buffer_len
                            f.write(f"DEBUG FILE: Calculated average (sum/len) = {calculated_avg:.6f}\n")
                        else:
                            f.write("DEBUG FILE WARNING: Buffer length is zero inside 'if self._train_loss_buffer' block.\n")
                    except Exception as e_calc:
                         f.write(f"DEBUG FILE ERROR: Error during sum/avg calculation: {e_calc}\n")
                else:
                     f.write("DEBUG FILE WARNING: self._train_loss_buffer is empty or evaluates to False.\n")
            # === END: Write to debug file ===

            # --- Original logic ---
            if not self._train_loss_buffer:
                self.avg_train_loss = 0.0
                logging.info(f"Epoch {epoch} ended. Assigning avg_train_loss = 0.0 because buffer was empty.") 
            else:
                # Assign the average
                try:
                    buffer_len = len(self._train_loss_buffer)
                    if buffer_len > 0:
                        self.avg_train_loss = sum(self._train_loss_buffer) / buffer_len
                        logging.info(f"Epoch {epoch} ended. Assigning Average train loss: {self.avg_train_loss:.4f}") 
                    else:
                        logging.warning(f"Epoch {epoch}: Reached 'else' block but buffer length is {buffer_len}. Assigning 0.0.") 
                        self.avg_train_loss = 0.0
                except Exception as assign_e:
                     logging.error(f"Epoch {epoch}: Error calculating/assigning avg_train_loss in 'else' block: {assign_e}. Assigning 0.0.", exc_info=True) # Keep ERROR log
                     self.avg_train_loss = 0.0

            # --- Final check log (write to file) ---
            with open(debug_file_path, "a") as f:
                f.write(f"DEBUG FILE: Final value of self.avg_train_loss before clearing buffer = {self.avg_train_loss}\n")

            # --- Original buffer clearing ---
            self._train_loss_buffer = []
            with open(debug_file_path, "a") as f:
                f.write(f"DEBUG FILE: Cleared _train_loss_buffer for epoch {epoch}.\n")
            logging.info(f"Debug file written to {debug_file_path}") # Log that the file *should* have been written

        except Exception as e:
             # Log the main exception normally
             logging.error(f"Error in on_train_epoch_end for epoch {epoch}: {e}", exc_info=True)
             # Also write to debug file
             try:
                 with open(debug_file_path, "a") as f:
                     f.write(f"DEBUG FILE CRITICAL ERROR in on_train_epoch_end: {e}\n")
             except Exception: pass # Avoid errors during error handling
             self.avg_train_loss = 0.0
             self._train_loss_buffer = []
             try:
                 with open(debug_file_path, "a") as f:
                     f.write(f"DEBUG FILE: Cleared _train_loss_buffer after CRITICAL ERROR for epoch {epoch}.\n")
             except Exception: pass

    # --- on_evaluate_end (Using getattr for hparams access) ---
    def on_evaluate_end(self, avg_valid_loss, valid_stats):
        """Called after validation loop to check scheduler"""
        # ***** Use getattr for hparams access *****
        lr_scheduler = getattr(self.hparams, "lr_scheduler", None)
        if (lr_scheduler and hasattr(self, 'optimizer') and self.optimizer and
            valid_stats and "WER" in valid_stats):
            try:
                 wer_value = valid_stats["WER"]
                 if isinstance(wer_value, (int, float)) and math.isfinite(wer_value):
                      old_lr, new_lr = lr_scheduler(metric_value=wer_value) # Call scheduler instance
                      if new_lr != old_lr:
                           sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
                           logging.info(f"LR updated via scheduler: {old_lr:.6e} -> {new_lr:.6e}")
                 else:
                      logging.warning(f"Invalid WER ({wer_value}) for LR scheduling.")
            except Exception as e:
                 logging.error(f"Error during LR scheduler step/update: {e}")
        else:
             # Log which component is missing
             missing = []
             if not lr_scheduler: missing.append("LR scheduler")
             if not (hasattr(self, 'optimizer') and self.optimizer): missing.append("Optimizer")
             if not (valid_stats and "WER" in valid_stats): missing.append("Valid WER metric")
             logging.info(f"Skipping LR update. Missing: {', '.join(missing)}")


print("WhisperFineTuneBrain defined.")
# End of Cell 4


# Cell 5: Define Main Modal Training Function

logging.info("Defining Modal training function...")

@app.function(
    image=modal_image,
    gpu="A100-40GB",  # Change this in case you want to use a different GPU
    cpu=8,
    volumes={CHECKPOINT_DIR: volume},
    secrets=[hf_secret, wandb_secret],
    timeout=7200
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
            project_name = getattr(hparams, "wandb_project", "project-name") # Placeholder : Change this to your project's name on weights and biases
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
            import speechbrain as sb
            logging.info(f"SpeechBrain version: {sb.__version__}")

            logging.info("Importing datasets...")
            from datasets import load_dataset, DatasetDict, Audio, concatenate_datasets
            logging.info("Datasets library imported successfully")

            logging.info("Importing SpeechBrain components...")
            from speechbrain.dataio.dataset import DynamicItemDataset
            from speechbrain.dataio.dataloader import SaveableDataLoader
            from speechbrain.dataio.batch import PaddedBatch
            from speechbrain.dataio.sampler import DynamicBatchSampler
            from speechbrain.utils.distributed import run_on_main, ddp_init_group
            # Revert logger import to only FileTrainLogger
            from speechbrain.utils.train_logger import FileTrainLogger 
            from speechbrain.lobes.models.huggingface_transformers.whisper import Whisper
            logging.info("SpeechBrain components imported successfully")

            logging.info("Importing optimization libraries...")
            import torch.optim as optim
            import pandas as pd
            import math
            logging.info("Optimization libraries imported successfully")

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
                    array = audio_data.get("array")
                    sampling_rate = audio_data.get("sampling_rate")
                    if array is not None and sampling_rate is not None and sampling_rate > 0:
                        duration = len(array) / sampling_rate
                batch["duration"] = duration
            except Exception as e: logging.warning(f"Could not calculate duration: {e}. Setting to {duration}."); batch["duration"] = duration
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
        whisper_model = modules = optimizer = lr_scheduler = None
        try:
            logging.info("Initializing model, optimizer, scheduler...")
            whisper_model = Whisper(
                source=hparams.get("whisper_hub"), save_path=save_folder, encoder_only=False,
                language=hparams.get("language"), task=hparams.get("task")
            )
            modules = {"whisper": whisper_model}
            no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
            params = [
                {'params': [p for n, p in modules["whisper"].named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
                 'weight_decay': hparams.get("weight_decay", 0.05), 'lr': hparams.get("learning_rate", 1e-7)},
                {'params': [p for n, p in modules["whisper"].named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': hparams.get("learning_rate", 1e-7)}
            ]
            optimizer = optim.AdamW(
                params=params,
                lr=hparams.get("learning_rate", 1e-7),
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=hparams.get("weight_decay", 0.05)
            )
            lr_scheduler = sb.nnet.schedulers.NewBobScheduler(
                initial_value=hparams.get("learning_rate"), improvement_threshold=hparams.get("lr_improvement_threshold", 0.0025),
                annealing_factor=hparams.get("lr_annealing_factor"), patient=hparams.get("lr_patient", 0)
            )
            logging.info("Model, optimizer, scheduler initialized.")
        except KeyError as e: logging.error(f"Missing critical hparam for setup: {e}"); return
        except Exception as e: logging.error(f"Error initializing model/optimizer/scheduler: {e}", exc_info=True); return

        # --- Checkpointer and Logger Setup --- 
        epoch_counter = checkpointer = train_logger = None
        try:
            epoch_counter = sb.utils.epoch_loop.EpochCounter(limit=hparams.get("epochs"))
            checkpointer = sb.utils.checkpoints.Checkpointer(
                checkpoints_dir=save_folder,
                recoverables={ "model": modules["whisper"], "scheduler": lr_scheduler, "counter": epoch_counter, "optimizer": optimizer, },
            )

            # --- Use Unconditional FileTrainLogger --- 
            train_logger = FileTrainLogger(save_file=os.path.join(output_folder, "train_log.txt"))
            hparams["train_logger"] = train_logger # Assign the logger
            logging.info("Initialized FileTrainLogger.")

        except KeyError as e: 
            logging.error(f"Missing critical hparam for checkpointer/logger setup: {e}")

        # --- Create Datasets ---
        logging.info("Creating SpeechBrain DynamicItemDatasets...")
        datasets_dict = {}
        output_keys = ["id", "signal_raw", "text_raw", "duration"]
        try:
            required_splits = [hparams.get("train_split"), hparams.get("valid_split"), hparams.get("test_split")]
            required_splits = [s for s in required_splits if s]
            if not required_splits: logging.error("No dataset split names defined in hparams."); return
            for split in required_splits:
                if split in raw_datasets:
                    dynamic_items = [audio_pipeline_minimal, text_pipeline_minimal]
                    hf_dataset_split = raw_datasets[split]
                    data_dict = {str(i): hf_dataset_split[i] for i in range(len(hf_dataset_split))}
                    datasets_dict[split] = DynamicItemDataset(
                         data=data_dict, dynamic_items=dynamic_items, output_keys=output_keys,
                    )
                    logging.info(f"Successfully created DynamicItemDataset for split: {split} with {len(datasets_dict[split])} items.")
                else:
                     logging.warning(f"Split '{split}' not found in loaded dataset. Skipping.")
            if not hparams.get("train_split") in datasets_dict or not hparams.get("valid_split") in datasets_dict:
                 logging.error("Essential train or validation dataset could not be created. Exiting.")
                 return
        except Exception as e:
             logging.error(f"Error creating DynamicItemDatasets: {e}", exc_info=True)
             return

        # --- Dataloader Kwargs and Samplers ---
        logging.info("Creating Dataloaders with DynamicBatchSampler...")
        train_loader_kwargs = valid_loader_kwargs = test_loader_kwargs = None
        try:
            dynamic_batching = hparams.get("batch_size_dynamic", True)
            loader_common_kwargs = {
                "num_workers": hparams.get("num_workers", 0),
                "pin_memory": True if hparams.get("num_workers", 0) > 0 else False,
                "prefetch_factor": 2 if hparams.get("num_workers", 0) > 0 else None,
                "collate_fn": PaddedBatch,
            }
            if dynamic_batching:
                max_batch_length_samples = int(hparams.get("max_batch_len_seconds") * hparams.get("target_sample_rate"))
                # Get num_buckets from hparams
                num_buckets = hparams.get("dynamic_batch_num_buckets", 60) # Default to 60 if not set
                logging.info(f"Using dynamic batching with max_batch_length={max_batch_length_samples} samples and num_buckets={num_buckets}.")

                def length_func(item_dict):
                    duration = item_dict.get("duration")
                    if duration is None or not isinstance(duration, (int, float)) or duration < 0: return 0
                    return math.ceil(duration * hparams.get("target_sample_rate"))
                train_split_name = hparams.get("train_split")
                if train_split_name and train_split_name in datasets_dict:
                    train_sampler = DynamicBatchSampler(
                        datasets_dict[train_split_name],
                        max_batch_length=max_batch_length_samples,
                        num_buckets=num_buckets, # Use hparam value
                        shuffle=True,
                        batch_ordering="random",
                        length_func=length_func
                    )
                    train_loader_kwargs = { **loader_common_kwargs, "batch_sampler": train_sampler, "shuffle": False }
                if hparams.get("valid_split") in datasets_dict:
                    valid_sampler = DynamicBatchSampler(
                        datasets_dict[hparams.get("valid_split")],
                        max_batch_length=max_batch_length_samples,
                        num_buckets=num_buckets, # Use hparam value
                        shuffle=False,
                        batch_ordering="random",
                        length_func=length_func
                    )
                    valid_loader_kwargs = { **loader_common_kwargs, "batch_sampler": valid_sampler, "shuffle": False }
                if hparams.get("test_split") in datasets_dict:
                    test_sampler = DynamicBatchSampler(
                        datasets_dict[hparams.get("test_split")],
                        max_batch_length=max_batch_length_samples,
                        num_buckets=num_buckets, # Use hparam value
                        shuffle=False,
                        batch_ordering="random",
                        length_func=length_func
                    )
                    test_loader_kwargs = { **loader_common_kwargs, "batch_sampler": test_sampler, "shuffle": False }
            else:
                static_bs = hparams.get("loader_batch_size", 8)
                loader_common_kwargs["batch_size"] = static_bs
                if hparams.get("train_split") in datasets_dict:
                    train_loader_kwargs = { **loader_common_kwargs, "shuffle": True }
                if hparams.get("valid_split") in datasets_dict:
                    valid_loader_kwargs = { **loader_common_kwargs, "shuffle": False }
                if hparams.get("test_split") in datasets_dict:
                    test_loader_kwargs = { **loader_common_kwargs, "shuffle": False }
            if not train_loader_kwargs or not valid_loader_kwargs:
                logging.error("Essential train or validation loader could not be created. Exiting.")
                return
        except KeyError as e: logging.error(f"Missing critical hparam for dataloader setup: {e}"); return
        except Exception as e: logging.error(f"Error creating Dataloaders/Samplers: {e}", exc_info=True); return

        # --- Initialize Brain ---
        whisper_brain = None
        try:
            logging.info("Initializing WhisperFineTuneBrain...")
            if lr_scheduler: hparams['lr_scheduler'] = lr_scheduler
            if epoch_counter: hparams['epoch_counter'] = epoch_counter
            hparams['train_logger'] = train_logger if train_logger else None
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            run_opts = {"device": device_type}
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                run_opts.update({
                    "data_parallel_backend": "ddp",
                    "ddp_port": os.environ.get("MASTER_PORT", "29500"),
                    "ddp_init_method": "env://",
                })
            whisper_brain = WhisperFineTuneBrain(
                modules=modules, opt_class=lambda params: optimizer, hparams=hparams,
                run_opts=run_opts, checkpointer=checkpointer
            )
            logging.info(f"WhisperFineTuneBrain initialized on device: {whisper_brain.device}")
        except Exception as e: logging.error(f"Error initializing WhisperFineTuneBrain: {e}", exc_info=True); return

        # --- WandB Watch Model (If enabled and wandb initialized) ---
        # Check if wandb_run exists (meaning wandb.init was successful)
        if wandb_run and hparams.get("wandb_watch_model", False):
             try:
                 import wandb 
                 # Use .get() for watch frequency too
                 watch_freq = hparams.get("wandb_watch_freq", 100) # Placeholder : Change this to your desired watch frequency
                 # Watch the specific model module within the Brain class
                 wandb.watch(whisper_brain.modules.whisper, log="gradients", log_freq=watch_freq)
                 logging.info(f"WandB watching model gradients every {watch_freq} steps.")
             except ImportError:
                  logging.warning("wandb library not found, cannot watch model.")
             except Exception as e:
                 logging.warning(f"Could not set up wandb.watch: {e}")
        else:
             missing_watch_reasons = []
             if not wandb_run: missing_watch_reasons.append("W&B run not initialized")
             if not hparams.get("wandb_watch_model", False): missing_watch_reasons.append("W&B watch disabled in hparams")
             if missing_watch_reasons: logging.info(f"Skipping wandb.watch because: {', '.join(missing_watch_reasons)}")

        # --- Training ---
        train_set = datasets_dict.get(hparams.get("train_split"))
        valid_set = datasets_dict.get(hparams.get("valid_split"))
        if train_set and valid_set and train_loader_kwargs and valid_loader_kwargs and epoch_counter and whisper_brain:
            logging.info(f"Starting training loop for {getattr(epoch_counter, 'limit', 'N/A')} epochs...")
            try:
                if checkpointer:
                    try: checkpointer.recover_if_possible(); logging.info("Attempted checkpoint recovery.")
                    except Exception as ckpt_load_e: logging.warning(f"Could not recover checkpoint before fit: {ckpt_load_e}")
                whisper_brain.fit(
                    epoch_counter=epoch_counter, train_set=train_set, valid_set=valid_set,
                    train_loader_kwargs=train_loader_kwargs, valid_loader_kwargs=valid_loader_kwargs
                )
                logging.info("Training complete.")
                try: volume.commit(); logging.info("Volume committed after training.")
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