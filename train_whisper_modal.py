import logging
import datetime
import yaml  # Add yaml import
import tempfile  # Add tempfile for safe temporary file handling
import csv  # Add csv import for augmentation file handling
import shutil  # For cleanup
import math
import torch.nn.functional as F
from speechbrain.processing.signal_processing import compute_amplitude, dB_to_amplitude, reverberate
from datasets import load_dataset  # Global import for dataset loading in Brain

# Turn off tokenizers parallelism warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Full Corrected Script V4 (Fix SimpleNamespace error, Add Gradient Clipping)

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
app = modal.App("speechbrain-whisper-finetune-egyptian")

# --- Secrets ---
hf_secret = modal.Secret.from_name("huggingface-secret-write")
wandb_secret = modal.Secret.from_name("wandb-secret") # Define the wandb secret object

# --- Persistent Storage ---
volume = modal.Volume.from_name(
    "speechbrain-finetune-storage", create_if_missing=True
)
CACHE_DIR = "/cache" # HF cache inside container
CHECKPOINT_DIR = "/root/checkpoints" # Mount point inside container

# --- Environment Image Definition ---
modal_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "build-essential", "cmake", "libboost-all-dev",
        "libeigen3-dev", "git", "libsndfile1", "ffmpeg"
    )
    .pip_install(
        "pip==23.3.2",
        "setuptools==69.0.3",
        "wheel==0.42.0",
        "pyarrow==15.0.0",      # Pin pyarrow *before* datasets (Check compatibility!)
        # Core ML libraries
        "torch==2.1.2",  # Latest stable that's well-tested with Whisper
        "torchaudio==2.1.2",  # Matching torch version
        "torchvision==0.16.2", # Add torchvision, compatible with torch 2.1.2
        "transformers==4.51.3",  # Use latest
        "accelerate==0.25.0",  # Latest stable
        "wandb",  # <-- ADD WANDB HERE
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
    "hf_dataset_id": "MAdel121/arabic-egy-cleaned",
    "train_split": "train",
    "valid_split": "validation",
    "test_split": "test",
    "noise_dataset_id": "Myrtle/CAIMAN-ASR-BackgroundNoise",
    "rir_dataset_id": "Fhrozen/tau_srir_db",
    "target_sample_rate": TARGET_SAMPLE_RATE,

    # Model & Tokenizer
    "whisper_hub": "openai/whisper-small",
    "save_folder": f"{CHECKPOINT_DIR}/whisper_small_egy_save",
    "output_folder": f"{CHECKPOINT_DIR}/whisper_small_egy_output",
    "language": "ar",
    "task": "transcribe",

    # Token IDs (Updated from verification using Processor)
    "sot_index": token_info["sot_token_id"] if token_info else 50258,
    "eos_index": token_info["eos_token_id"] if token_info else 50257,
    "pad_token_id": token_info["pad_token_id"] if token_info else 50257,
    "decoder_start_ids": token_info["decoder_start_ids"] if token_info else [50258, 50361, 50359, 50363],

    # Augmentation Params
    "augment": True,
    "noise_prob": 0.50,  # Increased
    "reverb_prob": 0.30, # Increased
    "speed_prob": 0.50,  # Increased
    "pitch_prob": 0.30,  # Increased (Manual)
    "gain_prob": 0.50,   # Increased (Manual)
    "min_augmentations": 1,
    "max_augmentations": 1,  # Increased - Allow combinations
    "noise_snr_low": 15,    # Keep severity mild for now
    "noise_snr_high": 25,
    "speed_factors": [95, 105], # Keep severity mild for now
    "pitch_steps_low": -1,  # Keep severity mild for now
    "pitch_steps_high": 1,
    "gain_db_low": -4,     # Keep severity mild for now
    "gain_db_high": 4,

    # Training Params
    "seed": 1986,
    "epochs": 5,
    "learning_rate": 5e-7,
    "lr_warmup_steps": 1000,
    "weight_decay": 0.05,
    "lr_annealing_factor": 0.9,
    "batch_size_dynamic": False,
    "loader_batch_size": 8,
    "max_batch_len_seconds": 5.0,
    "num_workers": 4,
    "grad_accumulation_factor": 2,
    "max_grad_norm": 5.0,

    # Checkpointing
    "ckpt_interval_minutes": 30,
    "num_checkpoints_to_keep": 2,

    # Whisper decoding params
    "num_beams": 5,

    # === W&B Configuration ===
    "use_wandb": True,  # Flag to enable/disable easily
    "wandb_project": "Whisper-Egyptian-Finetune",  # Project name
    "wandb_entity": None,  # Optional: Your W&B username or team name
    "wandb_log_batch_freq": 100,  # Log batch metrics every N steps
    "wandb_watch_model": True,  # Whether to watch gradients
    "wandb_watch_freq": 100,  # How often to log gradients
}

# Add a check after hparams definition
if not token_info:
    logging.warning("Failed to verify token IDs. Using default values in hparams, which might be incorrect.")
else:
    logging.info("Successfully verified token IDs and updated hparams.")

print("Hyperparameters defined.")
# End of Cell 2


# Cell 3: Define Data Loading/Preprocessing (Unchanged from V3)

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

# --- Manual Augmentation Functions (will be called inside Brain - unchanged) ---
def _apply_pitch_shift(waveform, sr, prob, low, high):
    if random.random() < prob:
        n_steps = random.randint(low, high)
        if n_steps != 0:
            try:
                result = torchaudio.functional.pitch_shift(waveform.cpu(), sr, n_steps=n_steps).to(waveform.device)
                if not torch.all(torch.isfinite(result)):
                    logging.warning("NaN/Inf detected after pitch shift. Using original waveform.")
                    return waveform
                return result
            except Exception as e:
                logging.warning(f"torchaudio pitch shift failed: {e}")
    return waveform

def _apply_gain(waveform, prob, low_db, high_db):
    if random.random() < prob:
        gain_db = random.uniform(low_db, high_db)
        gain_amp = 10.0 ** (gain_db / 20.0) if gain_db > -float('inf') else 0.0
        result = waveform * gain_amp
        result = torch.clamp(result, min=-1.0, max=1.0)
        
        if not torch.all(torch.isfinite(result)):
            logging.warning("NaN/Inf detected after gain adjustment. Using original waveform.")
            return waveform
        return result
    return waveform

print("Data Loading Pipelines defined.")
# End of Cell 3


# Cell 4: Define Brain Subclass (Modified Augmentation Initialization)

import speechbrain as sb
from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.augment.time_domain import SpeedPerturb, AddNoise, AddReverb
from speechbrain.augment.augmenter import Augmenter
import os
import string
import torch
import torch.nn.utils.rnn as rnn_utils
import math

# --- Wrapper Augmentation Modules (placed before Brain class for visibility) ---

class NoiseSampler(torch.nn.Module):
    """Applies noise by sampling directly from an HF dataset object."""
    def __init__(self, noise_dataset, snr_low=0, snr_high=0, target_sample_rate=16000, normalize=False):
        super().__init__()
        self.noise_dataset = noise_dataset
        self.num_noises = len(noise_dataset) if noise_dataset is not None else 0
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.target_sample_rate = target_sample_rate
        self.normalize = normalize

    def forward(self, waveforms, lengths):
        if self.num_noises == 0:
            return waveforms

        batch_size, max_len = waveforms.shape[0], waveforms.shape[1]
        abs_lengths = (lengths * max_len).long().squeeze(1) if lengths.dim() == 2 else (lengths * max_len).long()

        clean_amp = compute_amplitude(waveforms, abs_lengths.unsqueeze(1), amp_type="rms")
        SNR = torch.rand(batch_size, 1, device=waveforms.device) * (self.snr_high - self.snr_low) + self.snr_low
        noise_factor = 1 / (dB_to_amplitude(SNR) + 1)
        if waveforms.dim() == 3:
            noise_factor = noise_factor.unsqueeze(1)

        new_noise_amp = noise_factor * clean_amp
        noisy_waveform = waveforms * (1 - noise_factor)

        processed = []
        for i in range(batch_size):
            try:
                idx = random.randint(0, self.num_noises - 1)
                item = self.noise_dataset[idx]["audio"]
                noise_wav = torch.tensor(item["array"], device=waveforms.device).float()
                sr = item["sampling_rate"]
                if sr != self.target_sample_rate:
                    noise_wav = torchaudio.functional.resample(noise_wav, sr, self.target_sample_rate)

                tgt_len = abs_lengths[i].item()
                cur_len = noise_wav.shape[-1]
                if cur_len > tgt_len:
                    start = random.randint(0, cur_len - tgt_len)
                    noise_wav = noise_wav[..., start:start+tgt_len]
                elif cur_len < tgt_len:
                    noise_wav = torch.nn.functional.pad(noise_wav, (0, tgt_len - cur_len))

                orig_amp = compute_amplitude(noise_wav.unsqueeze(0), torch.tensor([noise_wav.numel()], device=waveforms.device), amp_type="rms")
                noise_wav = noise_wav * new_noise_amp[i] / (orig_amp + 1e-14)

                if noise_wav.shape[-1] < max_len:
                    noise_wav = torch.nn.functional.pad(noise_wav, (0, max_len - noise_wav.shape[-1]))

                processed.append(noise_wav)
            except Exception as e:
                logging.warning(f"NoiseSampler error on sample {i}: {e}")
                processed.append(torch.zeros(max_len, device=waveforms.device))

        noise_batch = torch.stack(processed, dim=0)
        noisy_waveform = noisy_waveform + noise_batch

        if self.normalize:
            abs_max, _ = torch.max(torch.abs(noisy_waveform), dim=1, keepdim=True)
            noisy_waveform = noisy_waveform / abs_max.clamp(min=1.0)

        return noisy_waveform


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

        # Create temporary directory for CSV files
        temp_dir = tempfile.mkdtemp()
        noise_csv_path = os.path.join(temp_dir, "noise_augment.csv")
        rir_csv_path = os.path.join(temp_dir, "rir_augment.csv")
        self._temp_dir = temp_dir  # Store for cleanup

        # ***** Use getattr for hparams access *****
        if getattr(self.hparams, "augment", False):
            sb_augmentations = []  # List to hold augmenter instances

            target_sr = getattr(self.hparams, "target_sample_rate", TARGET_SAMPLE_RATE)

            # --- Load Noise/RIR Dataset Objects ---
            hparams["noise_dataset_object"] = None
            hparams["rir_dataset_object"] = None
            if hparams.get("augment", False):
                try:
                     noise_ds_id = hparams.get("noise_dataset_id")
                     if noise_ds_id:
                          logging.info(f"Loading noise dataset object: {noise_ds_id}")
                          try:
                               noise_ds = load_dataset(noise_ds_id, cache_dir=CACHE_DIR, split="train")
                               hparams["noise_dataset_object"] = noise_ds
                               logging.info(f"Loaded noise dataset object with {len(noise_ds)} samples.")
                          except Exception as e_noise_load: logging.warning(f"Could not load noise dataset object {noise_ds_id}: {e_noise_load}")

                     rir_ds_id = hparams.get("rir_dataset_id")
                     if rir_ds_id:
                          logging.info(f"Loading RIR dataset object: {rir_ds_id}")
                          try:
                               rir_ds = load_dataset(rir_ds_id, cache_dir=CACHE_DIR, split="train")
                               hparams["rir_dataset_object"] = rir_ds
                               logging.info(f"Loaded RIR dataset object with {len(rir_ds)} samples.")
                          except Exception as e_rir_load: logging.warning(f"Could not load RIR dataset object {rir_ds_id}: {e_rir_load}")
                except Exception as e: logging.warning(f"General error loading noise/RIR dataset objects: {e}.")
            else: logging.info("Skipping noise/RIR dataset loading as augmentation is disabled.")

            # --- Initialize NoiseSampler using dataset object ---
            noise_dataset = getattr(self.hparams, "noise_dataset_object", None)
            noise_prob = getattr(self.hparams, "noise_prob", 0.0)
            if noise_prob > 0 and noise_dataset is not None:
                try:
                    noise_sampler = NoiseSampler(
                        noise_dataset=noise_dataset,
                        snr_low=self.hparams.noise_snr_low,
                        snr_high=self.hparams.noise_snr_high,
                        target_sample_rate=target_sr
                    )
                    sb_augmentations.append(noise_sampler)
                    initialized_augmentations.append("NoiseSampler")
                except Exception as e:
                    logging.warning(f"Could not initialize NoiseSampler: {e}. Skipping.", exc_info=True)

            # --- Initialize RIRSampler using dataset object ---
            rir_dataset = getattr(self.hparams, "rir_dataset_object", None)
            reverb_prob = getattr(self.hparams, "reverb_prob", 0.0)
            if reverb_prob > 0 and rir_dataset is not None:
                try:
                    reverb_sampler = RIRSampler(
                        rir_dataset=rir_dataset,
                        rir_scale_factor=1.0,
                        target_sample_rate=target_sr
                    )
                    sb_augmentations.append(reverb_sampler)
                    initialized_augmentations.append("RIRSampler")
                except Exception as e:
                    logging.warning(f"Could not initialize RIRSampler: {e}. Skipping.", exc_info=True)

            # --- Initialize SpeedPerturb ---
            speed_prob = getattr(self.hparams, "speed_prob", 0.0)  # Get probability but don't use it
            if speed_prob > 0 and hasattr(self.hparams, "speed_factors"):
                try:
                    speeds = [factor / 100.0 for factor in self.hparams.speed_factors]
                    # Initialize WITHOUT perturb_prob (will use default 1.0)
                    speed_perturber = SpeedPerturb(
                        orig_freq=target_sr,
                        speeds=speeds,
                        device=self.device  # Added device parameter
                    )
                    sb_augmentations.append(speed_perturber)
                    initialized_augmentations.append("SpeedPerturb")
                except Exception as e:
                    logging.warning(f"Could not initialize SpeedPerturb: {e}. Skipping.", exc_info=True)

            # --- Initialize Augmenter (WITHOUT probs argument) ---
            if sb_augmentations:
                try:
                    # Initialize based on Augmenter source code (NO probs argument)
                    self.augmenter = Augmenter(
                        parallel_augment=False,
                        concat_original=False,
                        min_augmentations=getattr(self.hparams, "min_augmentations", 1),
                        max_augmentations=getattr(self.hparams, "max_augmentations", len(sb_augmentations)),
                        shuffle_augmentations=True,
                        augment_prob=1.0,
                        augmentations=sb_augmentations, # Pass list of sampler/perturber objects
                    )
                    aug_names = [type(aug).__name__ for aug in sb_augmentations]
                    logging.info(f"SpeechBrain Augmenter initialized with: {', '.join(aug_names)}")
                except Exception as e:
                    logging.warning(f"Could not initialize main Augmenter: {e}. Not initialized.", exc_info=True)
                    self.augmenter = None
                    initialized_augmentations = []
            else:
                logging.info("No SpeechBrain augmentations were added.")
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

            # ***** Apply SpeechBrain Augmenter to the whole batch *****
            if stage == sb.Stage.TRAIN and getattr(self.hparams, "augment", False) and self.augmenter is not None:
                try:
                    # Note: Augmenter expects [batch, time, channels] or [batch, time]
                    # Ensure signals has the right shape if needed (might need unsqueeze/squeeze later)
                    # Pass the relative lengths tensor directly
                    wavs, wav_lens = self.augmenter(wavs, wav_lens)
                    logging.debug("Applied SpeechBrain Augmenter to batch.") # Optional debug log

                except Exception as aug_e:
                    logging.warning(f"Batch Augmentation failed: {aug_e}. Using original batch.", exc_info=True)
                    wavs = signals # Revert to original if augmentation fails
                    wav_lens = signal_lens


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
            wavs, wav_lens, tokens_bos_data, _, _, _ = self._preprocess_batch(batch, stage)
            tokens_bos, _ = tokens_bos_data

            if wavs is None or tokens_bos is None or wavs.numel() == 0 or tokens_bos.numel() == 0:
                 logging.error("Empty/None tensors from preprocessing in compute_forward.")
                 bsz = batch.batch_size if hasattr(batch, 'batch_size') else 1
                 dummy_log_probs = torch.zeros((bsz, 1, self.tokenizer.vocab_size), device=self.device)
                 dummy_wav_lens = torch.zeros(bsz, device=self.device)
                 return dummy_log_probs, None, dummy_wav_lens

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

            return log_probs, hyps_out, wav_lens.to(self.device)

        except Exception as e:
             logging.error(f"Error in compute_forward: {e}", exc_info=True)
             bsz = batch.batch_size if hasattr(batch, 'batch_size') else 1
             dummy_log_probs = torch.zeros((bsz, 1, self.tokenizer.vocab_size), device=self.device)
             dummy_wav_lens = torch.zeros(bsz, device=self.device)
             return dummy_log_probs, None, dummy_wav_lens


    # --- compute_objectives (Mostly unchanged, relies on preprocessing fix) ---
    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss and evaluation metrics."""
        try:
            log_probs, hyps, wav_lens = predictions

            if log_probs is None or not torch.all(torch.isfinite(log_probs)):
                 logging.error("Invalid log_probs (None, NaN/Inf) in compute_objectives. Returning zero loss.")
                 return torch.tensor(0.0, device=self.device, requires_grad=False)

            ids = getattr(batch, 'id', [f'no_id_{i}' for i in range(log_probs.shape[0])])

            try:
                 _, _, _, tokens_eos_data, tokens_data, target_words_raw = self._preprocess_batch(batch, stage)
                 if tokens_eos_data is None or tokens_eos_data[0] is None:
                      logging.error("Failed to retrieve targets from preprocessing.")
                      return torch.tensor(0.0, device=self.device, requires_grad=False)
                 tokens_eos_padded, tokens_eos_lens_rel = tokens_eos_data
                 tokens_padded, tokens_lens_rel = tokens_data
            except Exception as e:
                 logging.error(f"Error retrieving targets in compute_objectives: {e}")
                 return torch.tensor(0.0, device=self.device, requires_grad=False)

            if tokens_eos_padded is None or tokens_eos_padded.numel() == 0:
                logging.error("Empty target tensor in compute_objectives. Returning zero loss.")
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

    # --- on_stage_end (Summarize from SEPARATE objects) ---
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

                # --- RE-ADD Manual W&B logging for epoch stats ---
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

                # --- Checkpointing (Only on VALID stage) ---
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

    # --- on_fit_start (Updated with temporary variable) ---
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
                        # REMOVE logging the parameter name string
                        # if max_norm_name:
                        #     wandb_step_metrics["train/max_grad_param"] = max_norm_name
                        
                        # Only log finite numerical values
                        wandb.log({k: v for k, v in wandb_step_metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)})
                        
                        # REMOVE separate string logging
                        # str_metrics = {k: v for k, v in wandb_step_metrics.items() if isinstance(v, str)}
                        # if str_metrics:
                        #     wandb.log(str_metrics)
                            
                except Exception as wandb_log_e:
                    logging.warning(f"Could not log step metrics to W&B: {wandb_log_e}")

            # --- Accumulate loss for epoch average ---
            self._train_loss_buffer.append(batch_loss_value)

        except Exception as e:
            logging.error(f"Error in fit_batch: {e}", exc_info=True)
            loss = torch.tensor(float('nan'))  # Return NaN on error

        return loss.cpu() if isinstance(loss, torch.Tensor) else torch.tensor(loss)

    # --- on_train_epoch_end (WITH FILE DEBUGGING) ---
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
                logging.info(f"Epoch {epoch} ended. Assigning avg_train_loss = 0.0 because buffer was empty.") # Keep INFO log
            else:
                # Assign the average
                try:
                    buffer_len = len(self._train_loss_buffer)
                    if buffer_len > 0:
                        self.avg_train_loss = sum(self._train_loss_buffer) / buffer_len
                        logging.info(f"Epoch {epoch} ended. Assigning Average train loss: {self.avg_train_loss:.4f}") # Keep INFO log
                    else:
                        logging.warning(f"Epoch {epoch}: Reached 'else' block but buffer length is {buffer_len}. Assigning 0.0.") # Keep WARN log
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
    gpu="H100", 
    cpu=8,
    volumes={CHECKPOINT_DIR: volume},
    secrets=[hf_secret, wandb_secret],
    timeout=7200
)
def train_whisper_on_modal():
    global hparams # Keep global for now, though passing explicitly might be cleaner
    wandb_run = None # Initialize wandb_run variable

    # === Robust W&B Initialization (Manual Approach) ===
    if hparams.get("use_wandb", False):
        logging.info("Attempting W&B Initialization...")
        try:
            import wandb
            from datetime import datetime
            
            # Log check for API key environment variable
            wandb_api_key = os.environ.get("WANDB_API_KEY")
            if wandb_api_key:
                logging.info("WANDB_API_KEY environment variable found.")
                # Optionally log first few/last few chars for verification, but be careful!
                # logging.info(f"WANDB_API_KEY starts with: {wandb_api_key[:4]}") 
            else:
                 logging.warning("WANDB_API_KEY environment variable NOT found. W&B initialization might fail.")

            # Use standard string formatting to avoid f-string quote issues
            project_name = getattr(hparams, "wandb_project", "speechbrain-default")
            entity_name = hparams.get("wandb_entity", None) # Use .get()
            logging.info("Calling wandb.init with project='{}', entity='{}'".format(project_name, entity_name))
            
            wandb_run = wandb.init(
                project=project_name, # Use variable
                entity=entity_name, # Use variable
                config=hparams,
                name=f"whisper-egy-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                job_type="train",
                resume=False, # Explicitly set resume behavior if needed
            )
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
        # Avoid logging all env vars, can be too verbose and contain sensitive info
        # logging.info("Environment variables: %s", str(dict(os.environ)))

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
                def length_func(item_dict):
                    duration = item_dict.get("duration")
                    if duration is None or not isinstance(duration, (int, float)) or duration < 0: return 0
                    return math.ceil(duration * hparams.get("target_sample_rate"))
                train_split_name = hparams.get("train_split")
                if train_split_name and train_split_name in datasets_dict:
                    train_sampler = DynamicBatchSampler(datasets_dict[train_split_name], max_batch_length=max_batch_length_samples, num_buckets=100, shuffle=True, batch_ordering="random", length_func=length_func)
                    train_loader_kwargs = { **loader_common_kwargs, "batch_sampler": train_sampler, "shuffle": False }
                if hparams.get("valid_split") in datasets_dict:
                    valid_sampler = DynamicBatchSampler(datasets_dict[hparams.get("valid_split")], max_batch_length=max_batch_length_samples, num_buckets=100, shuffle=False, batch_ordering="random", length_func=length_func)
                    valid_loader_kwargs = { **loader_common_kwargs, "batch_sampler": valid_sampler, "shuffle": False }
                if hparams.get("test_split") in datasets_dict:
                    test_sampler = DynamicBatchSampler(datasets_dict[hparams.get("test_split")], max_batch_length=max_batch_length_samples, num_buckets=100, shuffle=False, batch_ordering="random", length_func=length_func)
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
                 # Ensure wandb is available
                 import wandb 
                 # Use .get() for watch frequency too
                 watch_freq = hparams.get("wandb_watch_freq", 100)
                 # Watch the specific model module within the Brain class
                 wandb.watch(whisper_brain.modules.whisper, log="gradients", log_freq=watch_freq)
                 logging.info(f"WandB watching model gradients every {watch_freq} steps.")
             except ImportError:
                  logging.warning("wandb library not found, cannot watch model.")
             except Exception as e:
                 logging.warning(f"Could not set up wandb.watch: {e}")
        else:
             # Log why watch wasn't enabled
             missing_watch_reasons = []
             if not wandb_run: missing_watch_reasons.append("W&B run not initialized")
             # Use .get() for the check here too
             if not hparams.get("wandb_watch_model", False): missing_watch_reasons.append("W&B watch disabled in hparams")
             # No need to check brain or logger type here if W&B isn't active
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
        if wandb_run: # Only finish if init was successful and wandb_run is assigned
            try:
                logging.info(f"Finishing W&B run: {wandb_run.name}")
                wandb.finish()
            except Exception as wandb_finish_e:
                logging.error(f"Error during wandb.finish(): {wandb_finish_e}")
        else:
            logging.info("W&B run was not active, skipping wandb.finish().")

        # --- Commit Volume --- 
        # This should happen regardless of W&B status
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