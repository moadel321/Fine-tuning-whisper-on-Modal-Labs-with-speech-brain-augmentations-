import logging
import datetime
import json
import os
import sys
import random
import glob
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Turn off tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Basic Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting NeMo ASR fine-tuning script...")

# Modal Setup
import modal

app = modal.App("fine-tune-nemo-arabic")

# Secrets
hf_secret = modal.Secret.from_name("huggingface-secret-write")
wandb_secret = modal.Secret.from_name("wandb-secret")

# Persistent Storage
volume = modal.Volume.from_name("fine-tune-nemo-arabic-volume", create_if_missing=True)
CHECKPOINT_DIR = "/root/checkpoints"
CACHE_DIR = f"{CHECKPOINT_DIR}/hf_cache"

# Environment Image
modal_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "build-essential", "cmake", "libboost-all-dev",
        "libeigen3-dev", "git", "libsndfile1", "ffmpeg", "wget"
    )
    .run_commands(
        "ln -sf /usr/bin/gcc /usr/bin/clang",
        "ln -sf /usr/bin/g++ /usr/bin/clang++"
    )
    .pip_install(
        "pip==23.3.2", "setuptools==69.5.1", "wheel==0.43.0", "cython>=3.0.11",
        "pybind11", "ninja", "packaging"
    )
    .pip_install(
        "torch==2.1.2", "torchaudio==2.1.2", "torchvision==0.16.2"
    )
    .pip_install(
        "numpy<2.0.0", "scipy>=1.10.0", "pandas>=2.0.0", 
        "tqdm>=4.65.0", "pyyaml==6.0.2"
    )
    .pip_install(
        "librosa>=0.10.0", "soundfile>=0.12.0", "sentencepiece>=0.1.99"
    )
    .pip_install(
        "transformers>=4.30.0", "datasets>=2.14.0", "huggingface_hub==0.16.4"
    )
    .pip_install(
        "hydra-core>=1.3.0", "omegaconf>=2.3.0", 
        "lhotse>=1.17.0", "einops>=0.7.0", "jiwer>=3.0.0", 
        "pyannote.core>=5.0.0", "editdistance>=0.6.0",
        "pyannote.metrics>=3.2.0", "webdataset>=0.2.86", "braceexpand>=0.1.7"
    )

    .pip_install(
        "onnx>=1.7.0", "wrapt", "ruamel.yaml", "wget", "frozendict", "unidecode",
        "black==19.10b0", "isort<5", "parameterized", "pytest", "pytest-runner",
        "sphinx", "sphinxcontrib-bibtex", "inflect"
    )
    .pip_install(
        "git+https://github.com/LahiLuk/YouTokenToMe.git", "text-unidecode>=1.3", "editdistance>=0.6.0",
        "kaldiio>=2.17.0", "kaldi-python-io>=1.2.0", "python-dateutil>=2.8.0",
        "marshmallow>=3.17.0", "webdataset>=0.2.86", "braceexpand>=0.1.7",
        "nemo-text-processing>=1.0.0"
    )
    .pip_install("pytorch-lightning==2.0.7")
    .pip_install("nemo_toolkit==1.23.0", "--no-deps")
    .pip_install("speechbrain>=0.5.15", "wandb>=0.15.0")
)

print("Modal setup complete.")

# Hyperparameters
TARGET_SAMPLE_RATE = 16000

hparams = {
    # Data
    "hf_dataset_id": "MAdel121/arabic-egy-cleaned",
    "train_split": "train",
    "valid_split": "validation", 
    "test_split": "test",
    "target_sample_rate": TARGET_SAMPLE_RATE,
    
    # Model
    "nemo_model_name": "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0",
    "save_folder": f"{CHECKPOINT_DIR}/nemo_arabic_save",
    "output_folder": f"{CHECKPOINT_DIR}/nemo_arabic_output",
    
    # Augmentation (only enabled ones)
    "augment": True,
    "num_augmented_variants": 2,  # Number of augmented versions per original
    "augment_prob": 0.7,  # Probability of applying augmentations
    
    # DropChunk params
    "drop_chunk_length_low": 1600,
    "drop_chunk_length_high": 4800,
    "drop_chunk_count_low": 1,
    "drop_chunk_count_high": 5,
    
    # DropFreq params  
    "drop_freq_count_low": 1,
    "drop_freq_count_high": 3,
    
    # Training
    "debug_mode": None, # Flag for test runs on a small data subset
    "debug_num_samples": 1000, # Number of samples for debug mode
    "seed": 1986,
    "epochs": 10,
    "learning_rate": 2e-5,  # Reduced from 1e-4 for stable fine-tuning
    "weight_decay": 0.005,  # Reduced from 0.01 for fine-tuning
    "batch_size": 8,
    "num_workers": 8,
    "grad_accumulation_factor": 4,
    "max_grad_norm": 5.0,
    
    # Scheduler parameters
    "scheduler_name": "CosineAnnealing",
    "min_lr": 1e-6,
    "warmup_ratio": 0.05,  # 5% warmup instead of 10%
    
    # W&B
    "use_wandb": True,
    "wandb_project": "nemo-arabic-asr",
    "wandb_entity": None,
    "wandb_resume_id": None,  # Set to specific run ID string to resume a W&B run
    
    # Additional metrics tracking
    "log_audio_samples": False,  # Log audio samples to W&B (can be large)
    "log_predictions": False,    # Log prediction examples
    "log_system_metrics": True,  # Log GPU/CPU usage metrics
    
    # Resume functionality
    "auto_resume": True,              # Automatically resume from latest checkpoint if found
    "force_restart": False,           # Force restart even if checkpoints exist
    "resume_checkpoint_path": None,   # Specific checkpoint path to resume from
    "skip_data_processing": True,     # Skip data processing if manifests exist
    "validate_data_integrity": False, # Validate existing data files before skipping
    "resume_strategy": "latest",      # "latest", "best", or "specific"
}

print("Hyperparameters defined.")

# Scheduler Calculation Functions
def calculate_scheduler_steps(dataset_size, batch_size, grad_accumulation, epochs, debug_mode=False):
    """Calculate total steps and appropriate warmup steps for scheduler."""
    effective_batch_size = batch_size * grad_accumulation
    steps_per_epoch = max(1, dataset_size // effective_batch_size)
    total_steps = steps_per_epoch * epochs
    
    # Dynamic warmup calculation based on mode
    if debug_mode:
        warmup_steps = min(50, total_steps // 10)  # Smaller for debug mode
    else:
        warmup_steps = min(500, int(total_steps * 0.05))  # 5% or 500 max
    
    return total_steps, warmup_steps, steps_per_epoch

# Augmentation Functions
def setup_augmentations():
    """Setup SpeechBrain augmentations for the enabled augmentations only."""
    from speechbrain.augment.time_domain import DropChunk, DropFreq, DropBitResolution
    from speechbrain.augment.augmenter import Augmenter
    
    augmentations = []
    
    # DropChunk
    drop_chunk = DropChunk(
        drop_length_low=hparams["drop_chunk_length_low"],
        drop_length_high=hparams["drop_chunk_length_high"],
        drop_count_low=hparams["drop_chunk_count_low"],
        drop_count_high=hparams["drop_chunk_count_high"],
    )
    augmentations.append(drop_chunk)
    
    # DropFreq
    drop_freq = DropFreq(
        drop_freq_count_low=hparams["drop_freq_count_low"],
        drop_freq_count_high=hparams["drop_freq_count_high"],
    )
    augmentations.append(drop_freq)
    
    # DropBitResolution
    drop_bit = DropBitResolution()
    augmentations.append(drop_bit)
    
    # Create augmenter
    augmenter = Augmenter(
        parallel_augment=False,
        concat_original=False,
        min_augmentations=1,
        max_augmentations=2,
        shuffle_augmentations=True,
        augment_prob=hparams["augment_prob"],
        augmentations=augmentations,
    )
    
    logging.info(f"Initialized augmenter with {len(augmentations)} augmentations")
    return augmenter

def apply_augmentations(audio_tensor, augmenter):
    """Apply augmentations to a single audio tensor."""
    try:
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        
        # Create relative lengths (all 1.0 since we're processing single files)
        lengths = torch.ones(audio_tensor.shape[0])
        
        # Apply augmentations
        augmented, _ = augmenter(audio_tensor, lengths)
        
        return augmented.squeeze(0)  # Remove batch dimension
    except Exception as e:
        logging.warning(f"Augmentation failed: {e}. Returning original audio.")
        return audio_tensor.squeeze(0) if audio_tensor.dim() > 1 else audio_tensor

def create_nemo_manifest(data_list, manifest_path):
    """Create NeMo manifest file from data list."""
    with open(manifest_path, 'w') as f:
        for item in data_list:
            f.write(json.dumps(item) + '\n')
    
    logging.info(f"Created manifest with {len(data_list)} entries: {manifest_path}")

def process_split(dataset, audio_dir, text_normalizer=None, augment_fn=None):
    """
    Processes a dataset split: saves audio to WAV, resamples, normalizes text,
    and optionally applies augmentations.

    Returns a list of dictionaries for the NeMo manifest.
    """
    os.makedirs(audio_dir, exist_ok=True)
    data_list = []

    for i, item in enumerate(dataset):
        try:
            audio_array = item['audio']['array']
            sample_rate = item['audio']['sampling_rate']
            text = item['text']

            # Resample audio if necessary
            audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
            if sample_rate != TARGET_SAMPLE_RATE:
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, sample_rate, TARGET_SAMPLE_RATE
                )
            
            # --- Original Audio ---
            original_path = os.path.join(audio_dir, f"original_{i:06d}.wav")
            sf.write(original_path, audio_tensor.numpy(), TARGET_SAMPLE_RATE)
            data_list.append({
                'audio_filepath': original_path,
                'duration': len(audio_tensor) / TARGET_SAMPLE_RATE,
                'text': text
            })

            # --- Augmented Audio (if augment_fn is provided) ---
            if augment_fn:
                augment_fn(audio_tensor, text, audio_dir, i, data_list)

            if (i + 1) % 250 == 0:
                logging.info(f"Processed {i + 1} audio files in split: {os.path.basename(audio_dir)}")

        except Exception as e:
            logging.error(f"Error processing item {i} in {os.path.basename(audio_dir)}: {e}")
            continue
            
    return data_list

# Resume functionality utilities
def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint file in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_pattern = os.path.join(checkpoint_dir, "*.ckpt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time, most recent first
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_files[0]

def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the best checkpoint based on validation metric."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for best checkpoint (PyTorch Lightning saves this)
    best_checkpoint = os.path.join(checkpoint_dir, "best.ckpt")
    if os.path.exists(best_checkpoint):
        return best_checkpoint
    
    # Fallback to latest if no best checkpoint found
    return find_latest_checkpoint(checkpoint_dir)

def validate_checkpoint_compatibility(checkpoint_path: str, current_config: dict) -> bool:
    """Check if checkpoint is compatible with current configuration."""
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Basic validation - check if it's a valid PyTorch Lightning checkpoint
        required_keys = ['epoch', 'global_step', 'state_dict']
        for key in required_keys:
            if key not in checkpoint:
                logging.warning(f"Checkpoint missing required key: {key}")
                return False
        
        return True
    except Exception as e:
        logging.error(f"Error validating checkpoint {checkpoint_path}: {e}")
        return False

def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """Extract metadata from checkpoint."""
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0),
            'path': checkpoint_path,
            'file_size': os.path.getsize(checkpoint_path),
            'modified_time': os.path.getmtime(checkpoint_path)
        }
        
        # Try to extract validation metrics if available
        if 'callbacks' in checkpoint:
            callbacks = checkpoint['callbacks']
            for callback_name, callback_state in callbacks.items():
                if 'best_model_score' in callback_state:
                    info['best_score'] = callback_state['best_model_score']
                    break
        
        return info
    except Exception as e:
        logging.error(f"Error getting checkpoint info from {checkpoint_path}: {e}")
        return {'path': checkpoint_path, 'error': str(e)}

def check_manifests_exist(manifest_dir: str) -> Dict[str, bool]:
    """Check which manifest files exist."""
    required_manifests = ['train_manifest.json', 'val_manifest.json', 'test_manifest.json']
    manifest_status = {}
    
    for manifest_name in required_manifests:
        manifest_path = os.path.join(manifest_dir, manifest_name)
        manifest_status[manifest_name] = os.path.exists(manifest_path)
    
    return manifest_status

def validate_manifest_integrity(manifest_path: str) -> bool:
    """Check if manifest file is valid JSON with expected structure."""
    try:
        with open(manifest_path, 'r') as f:
            line_count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse each line as JSON
                data = json.loads(line)
                
                # Check required fields
                required_fields = ['audio_filepath', 'duration', 'text']
                for field in required_fields:
                    if field not in data:
                        logging.warning(f"Manifest {manifest_path} missing field {field} in line {line_count + 1}")
                        return False
                
                line_count += 1
            
            # Must have at least one entry
            return line_count > 0
            
    except Exception as e:
        logging.error(f"Error validating manifest {manifest_path}: {e}")
        return False

def check_audio_files_exist(manifest_path: str) -> Tuple[bool, int, int]:
    """Check if audio files referenced in manifest exist. Returns (all_exist, found_count, total_count)."""
    try:
        found_count = 0
        total_count = 0
        
        with open(manifest_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                audio_path = data.get('audio_filepath')
                
                if audio_path:
                    total_count += 1
                    if os.path.exists(audio_path):
                        found_count += 1
        
        all_exist = found_count == total_count
        return all_exist, found_count, total_count
        
    except Exception as e:
        logging.error(f"Error checking audio files for manifest {manifest_path}: {e}")
        return False, 0, 0

def should_skip_data_processing(output_folder: str, hparams: dict) -> bool:
    """Determine if data processing should be skipped."""
    if not hparams.get("skip_data_processing", False):
        return False
    
    manifest_dir = os.path.join(output_folder, "manifests")
    manifest_status = check_manifests_exist(manifest_dir)
    
    # All manifests must exist
    if not all(manifest_status.values()):
        missing = [name for name, exists in manifest_status.items() if not exists]
        logging.info(f"Missing manifests: {missing}. Will process data.")
        return False
    
    # Validate manifest integrity
    for manifest_name, exists in manifest_status.items():
        if exists:
            manifest_path = os.path.join(manifest_dir, manifest_name)
            if not validate_manifest_integrity(manifest_path):
                logging.info(f"Invalid manifest: {manifest_name}. Will process data.")
                return False
    
    # Optionally validate audio files exist
    if hparams.get("validate_data_integrity", False):
        for manifest_name in manifest_status.keys():
            manifest_path = os.path.join(manifest_dir, manifest_name)
            all_exist, found, total = check_audio_files_exist(manifest_path)
            if not all_exist:
                logging.info(f"Missing audio files for {manifest_name}: {found}/{total} found. Will process data.")
                return False
    
    logging.info("All manifests exist and are valid. Skipping data processing.")
    return True

def validate_resume_setup(hparams: dict, resume_checkpoint_path: Optional[str], 
                         skip_data_processing: bool) -> bool:
    """Validate that resume setup is consistent and viable."""
    try:
        # Check for conflicting settings
        if hparams["force_restart"] and resume_checkpoint_path:
            logging.warning("force_restart=True but resume checkpoint found - this is expected")
        
        # If resuming, ensure we have valid manifests
        if resume_checkpoint_path and skip_data_processing:
            manifest_dir = os.path.join(hparams["output_folder"], "manifests")
            manifest_status = check_manifests_exist(manifest_dir)
            if not all(manifest_status.values()):
                logging.error("Resume checkpoint found but manifests are missing!")
                logging.error("This could cause training to fail. Consider setting skip_data_processing=False")
                return False
        
        # Validate scheduler parameters
        if hparams.get("learning_rate", 0) <= 0:
            logging.error(f"Invalid learning rate: {hparams.get('learning_rate')}")
            return False
        
        if hparams.get("min_lr", 0) >= hparams.get("learning_rate", 1e-4):
            logging.error(f"min_lr ({hparams.get('min_lr')}) should be less than learning_rate ({hparams.get('learning_rate')})")
            return False
        
        # Validate checkpoint directory exists
        if not os.path.exists(hparams["save_folder"]):
            logging.info(f"Checkpoint directory doesn't exist yet: {hparams['save_folder']}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating resume setup: {e}")
        return False

@app.function(
    image=modal_image,
    gpu="A100-40GB",
    cpu=8,
    volumes={CHECKPOINT_DIR: volume},
    secrets=[hf_secret, wandb_secret],
    timeout=8 * 60 * 60,
)
def train_nemo_arabic():
    """Main training function using NeMo framework."""
    
    # Setup environment
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
    os.environ["HF_HUB_CACHE"] = CACHE_DIR
    
    # Set seed
    torch.manual_seed(hparams["seed"])
    random.seed(hparams["seed"])
    
    # Setup directories
    os.makedirs(hparams["save_folder"], exist_ok=True)
    os.makedirs(hparams["output_folder"], exist_ok=True)
    
    # Resume detection logic
    resume_checkpoint_path = None
    skip_data_processing = False
    
    if not hparams["force_restart"] and hparams["auto_resume"]:
        logging.info("Checking for existing checkpoints...")
        
        if hparams["resume_checkpoint_path"]:
            # Specific checkpoint provided
            if os.path.exists(hparams["resume_checkpoint_path"]):
                if validate_checkpoint_compatibility(hparams["resume_checkpoint_path"], hparams):
                    resume_checkpoint_path = hparams["resume_checkpoint_path"]
                    logging.info(f"Will resume from specified checkpoint: {resume_checkpoint_path}")
                else:
                    logging.warning(f"Specified checkpoint is incompatible: {hparams['resume_checkpoint_path']}")
            else:
                logging.warning(f"Specified checkpoint not found: {hparams['resume_checkpoint_path']}")
        else:
            # Auto-detect checkpoint based on strategy
            if hparams["resume_strategy"] == "best":
                resume_checkpoint_path = find_best_checkpoint(hparams["save_folder"])
            else:  # "latest" or fallback
                resume_checkpoint_path = find_latest_checkpoint(hparams["save_folder"])
            
            if resume_checkpoint_path:
                if validate_checkpoint_compatibility(resume_checkpoint_path, hparams):
                    checkpoint_info = get_checkpoint_info(resume_checkpoint_path)
                    logging.info(f"Found valid checkpoint: {resume_checkpoint_path}")
                    logging.info(f"Checkpoint info: epoch={checkpoint_info.get('epoch', 'unknown')}, "
                               f"step={checkpoint_info.get('global_step', 'unknown')}")
                else:
                    logging.warning(f"Found checkpoint but it's incompatible: {resume_checkpoint_path}")
                    resume_checkpoint_path = None
    
    if hparams["force_restart"]:
        logging.info("Force restart enabled - ignoring any existing checkpoints")
        resume_checkpoint_path = None
    
    # Check if data processing should be skipped
    skip_data_processing = should_skip_data_processing(hparams["output_folder"], hparams)
    
    # Log resume summary
    logging.info("=" * 60)
    logging.info("RESUME FUNCTIONALITY SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Auto resume: {hparams['auto_resume']}")
    logging.info(f"Force restart: {hparams['force_restart']}")
    logging.info(f"Resume strategy: {hparams['resume_strategy']}")
    logging.info(f"Skip data processing: {hparams['skip_data_processing']}")
    
    if resume_checkpoint_path:
        logging.info(f"✓ RESUMING training from checkpoint: {resume_checkpoint_path}")
    else:
        logging.info("✓ STARTING fresh training")
    
    if skip_data_processing:
        logging.info("✓ SKIPPING data processing - using existing manifests")
    else:
        logging.info("✓ PROCESSING data - creating new manifests")
    logging.info("=" * 60)
    
    # Validate resume setup
    if not validate_resume_setup(hparams, resume_checkpoint_path, skip_data_processing):
        logging.warning("Resume setup validation failed - proceeding with caution")
    
    # Initialize W&B Logger
    wandb_logger = None
    if hparams["use_wandb"]:
        try:
            # Import WandbLogger here to handle import errors gracefully
            from pytorch_lightning.loggers import WandbLogger
            
            # Add scheduler info to config for W&B
            wandb_config = hparams.copy()
            wandb_config.update({
                'effective_batch_size': hparams['batch_size'] * hparams['grad_accumulation_factor'],
                'scheduler_info': 'Will be updated after dataset processing'
            })
            
            # Handle W&B run resuming
            wandb_logger_kwargs = {
                "project": hparams["wandb_project"],
                "entity": hparams["wandb_entity"],
                "config": wandb_config,
                "log_model": 'all',  # Log model checkpoints
                "save_dir": hparams["output_folder"]
            }
            
            # Check if we should resume an existing run
            resume_id = hparams.get("wandb_resume_id")
            if resume_id:
                logging.info(f"Attempting to resume W&B run with ID: {resume_id}")
                wandb_logger_kwargs.update({
                    "id": resume_id,
                    "resume": "must"  # Ensure it resumes or fails
                })
                # Don't set name when resuming - it will use the existing run's name
            else:
                logging.info("Starting a new W&B run")
                wandb_logger_kwargs["name"] = f"nemo-arabic-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            wandb_logger = WandbLogger(**wandb_logger_kwargs)
            # WandbLogger will automatically log:
            # - Training loss, validation loss
            # - WER (Word Error Rate), CER (Character Error Rate)  
            # - Learning rate, gradient norms
            # - Model checkpoints and hyperparameters
            logging.info(f"W&B Logger initialized: {wandb_logger.experiment.url}")
        except Exception as e:
            logging.warning(f"W&B Logger initialization failed: {e}")
            hparams["use_wandb"] = False
            wandb_logger = None
    
    try:
        # Import NeMo components
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        from datasets import load_dataset
        from omegaconf import open_dict
        
        logging.info("Loading dataset...")
        raw_datasets = load_dataset(hparams["hf_dataset_id"], cache_dir=CACHE_DIR)
        
        # --- Sub-sample for Debug Mode ---
        if hparams.get("debug_mode", False):
            logging.warning("--- DEBUG MODE ENABLED ---")
            debug_num_samples = hparams.get("debug_num_samples", 1000)
            # Use a smaller subset for validation and test
            debug_eval_samples = max(1, int(debug_num_samples * 0.1)) 
            logging.info(f"Using {debug_num_samples} train samples and {debug_eval_samples} eval samples.")

            for split_name in [hparams.get("train_split"), hparams.get("valid_split"), hparams.get("test_split")]:
                if split_name in raw_datasets:
                    num_samples = debug_num_samples if split_name == hparams.get("train_split") else debug_eval_samples
                    # Ensure we don't select more samples than exist in the original split
                    if len(raw_datasets[split_name]) > num_samples:
                        raw_datasets[split_name] = raw_datasets[split_name].select(range(num_samples))
            
            logging.info(f"Debug dataset sizes: { {s: len(d) for s, d in raw_datasets.items()} }")

        # Conditional data processing
        manifest_dir = os.path.join(hparams["output_folder"], "manifests")
        train_manifest = os.path.join(manifest_dir, "train_manifest.json")
        val_manifest = os.path.join(manifest_dir, "val_manifest.json")
        test_manifest = os.path.join(manifest_dir, "test_manifest.json")
        
        if not skip_data_processing:
            # Setup augmentations
            augmenter = setup_augmentations() if hparams["augment"] else None
            
            def augment_and_save(audio_tensor, text, audio_dir, index, data_list):
                """Applies augmentations and saves variants."""
                for variant in range(hparams["num_augmented_variants"]):
                    augmented_audio = apply_augmentations(audio_tensor, augmenter)
                    variant_path = os.path.join(audio_dir, f"augmented_{index:06d}_v{variant}.wav")
                    sf.write(variant_path, augmented_audio.numpy(), TARGET_SAMPLE_RATE)
                    data_list.append({
                        'audio_filepath': variant_path,
                        'duration': len(augmented_audio) / TARGET_SAMPLE_RATE,
                        'text': text
                    })

            # Create audio directories
            audio_base_dir = os.path.join(hparams["output_folder"], "audio")
            train_audio_dir = os.path.join(audio_base_dir, "train")
            val_audio_dir = os.path.join(audio_base_dir, "val")
            test_audio_dir = os.path.join(audio_base_dir, "test")
            
            # Process datasets and create manifests
            os.makedirs(manifest_dir, exist_ok=True)
            
            # Process training data
            logging.info("Processing training data...")
            train_data = process_split(
                raw_datasets[hparams["train_split"]],
                train_audio_dir,
                augment_fn=augment_and_save if hparams["augment"] and augmenter else None
            )
            create_nemo_manifest(train_data, train_manifest)
            
            # Process validation data (no augmentation)
            logging.info("Processing validation data...")
            val_data = process_split(
                raw_datasets[hparams["valid_split"]],
                val_audio_dir
            )
            create_nemo_manifest(val_data, val_manifest)
            
            # Process test data (no augmentation)
            logging.info("Processing test data...")
            test_data = process_split(
                raw_datasets[hparams["test_split"]],
                test_audio_dir
            )
            create_nemo_manifest(test_data, test_manifest)
        else:
            logging.info("Using existing manifest files:")
            logging.info(f"  Train: {train_manifest}")
            logging.info(f"  Validation: {val_manifest}")
            logging.info(f"  Test: {test_manifest}")
        
        # Load NeMo model
        if resume_checkpoint_path:
            logging.info(f"Loading NeMo model from checkpoint: {resume_checkpoint_path}")
            try:
                model = EncDecHybridRNNTCTCBPEModel.restore_from(resume_checkpoint_path)
                logging.info("Successfully loaded model from checkpoint")
            except Exception as e:
                logging.error(f"Failed to load model from checkpoint: {e}")
                logging.info("Falling back to pretrained model")
                model = EncDecHybridRNNTCTCBPEModel.from_pretrained(hparams["nemo_model_name"])
                resume_checkpoint_path = None  # Reset since we're not actually resuming
        else:
            logging.info("Loading pretrained NeMo model...")
            model = EncDecHybridRNNTCTCBPEModel.from_pretrained(hparams["nemo_model_name"])
        
        # Configure model for Arabic ASR and update paths
        with open_dict(model.cfg):
            # Always ensure these critical settings are correct
            model.cfg.use_cer = True
            model.cfg.tokenizer.dir = hparams["save_folder"]
            
            # Update dataset configs (always update paths in case they changed)
            model.cfg.train_ds.manifest_filepath = train_manifest
            model.cfg.train_ds.is_tarred = False
            model.cfg.train_ds.batch_size = hparams['batch_size']
            model.cfg.validation_ds.manifest_filepath = val_manifest
            model.cfg.validation_ds.batch_size = hparams['batch_size']
            model.cfg.test_ds.manifest_filepath = test_manifest
            model.cfg.test_ds.batch_size = hparams['batch_size']
            
            if resume_checkpoint_path:
                logging.info("Model loaded from checkpoint - preserving existing configuration where appropriate")
        
        # Log model configuration
        logging.info(f"Model labels count: {len(model.cfg.labels)}")
        logging.info(f"Model sample rate: {model.cfg.sample_rate}")
        logging.info(f"Using CER: {model.cfg.use_cer}")
        
        # Setup trainer configuration
        trainer_config = {
            'max_epochs': hparams["epochs"],
            'accelerator': 'gpu',
            'devices': 1,
            'gradient_clip_val': hparams["max_grad_norm"],
            'accumulate_grad_batches': hparams["grad_accumulation_factor"],
            'log_every_n_steps': 50,
            'val_check_interval': 1.0,  # Validate once per epoch for better stability
            'enable_checkpointing': True,
            'logger': wandb_logger if wandb_logger else False,
            'enable_progress_bar': True,
            'check_val_every_n_epoch': 1,  # Ensure validation runs every epoch
        }
        
        # Add resume configuration if resuming
        if resume_checkpoint_path:
            trainer_config['resume_from_checkpoint'] = resume_checkpoint_path
            logging.info(f"Trainer configured to resume from: {resume_checkpoint_path}")
        
        # Setup data configs as dictionaries (following NeMo tutorial pattern)
        logging.info("Setting up data loaders...")
        
        # Training data config
        train_ds = {}
        train_ds['manifest_filepath'] = [train_manifest]
        train_ds['sample_rate'] = hparams['target_sample_rate']
        train_ds['labels'] = model.cfg.labels  # Required field
        train_ds['batch_size'] = hparams['batch_size']
        train_ds['fused_batch_size'] = hparams['batch_size']  # Required for training
        train_ds['shuffle'] = True
        train_ds['max_duration'] = 20.0
        train_ds['min_duration'] = 0.5
        train_ds['pin_memory'] = True
        train_ds['is_tarred'] = False
        train_ds['num_workers'] = hparams['num_workers']
        
        # Validation data config
        validation_ds = {}
        validation_ds['sample_rate'] = hparams['target_sample_rate']
        validation_ds['manifest_filepath'] = [val_manifest]
        validation_ds['labels'] = model.cfg.labels  # Required field
        validation_ds['batch_size'] = hparams['batch_size']
        validation_ds['shuffle'] = False
        validation_ds['num_workers'] = hparams['num_workers']
        validation_ds['pin_memory'] = True
        validation_ds['use_cer'] = True  # Enable CER calculation for validation
        validation_ds['max_duration'] = 20.0  # Match training config
        validation_ds['min_duration'] = 0.5   # Match training config
        
        # Test data config
        test_ds = {}
        test_ds['sample_rate'] = hparams['target_sample_rate']
        test_ds['manifest_filepath'] = [test_manifest]
        test_ds['labels'] = model.cfg.labels  # Required field
        test_ds['batch_size'] = hparams['batch_size']
        test_ds['shuffle'] = False
        test_ds['num_workers'] = hparams['num_workers']
        test_ds['use_cer'] = True  # Use CER for evaluation
        
        # Setup data loaders
        model.setup_training_data(train_data_config=train_ds)
        model.setup_validation_data(val_data_config=validation_ds)
        model.setup_test_data(test_data_config=test_ds)
        
        # Log validation configuration
        logging.info("=" * 50)
        logging.info("VALIDATION CONFIGURATION")
        logging.info("=" * 50)
        logging.info(f"Validation manifest: {val_manifest}")
        logging.info(f"Validation batch size: {validation_ds['batch_size']}")
        logging.info(f"Use CER for validation: {validation_ds.get('use_cer', False)}")
        logging.info(f"Validation check interval: {trainer_config['val_check_interval']}")
        logging.info(f"Model compute_eval_loss: {getattr(model, 'compute_eval_loss', 'Not set')}")
        if hasattr(model, 'wer'):
            logging.info(f"Model WER use_cer: {getattr(model.wer, 'use_cer', 'Not set')}")
        logging.info("=" * 50)
        
        # Set model to training mode
        model.train()
        
        # Additional model configuration for evaluation and logging
        model.log_predictions = False
        model.compute_eval_loss = True  # Enable loss computation for logging
        
        # Configure WER/CER metrics for validation
        if hasattr(model, 'wer'):
            model.wer.use_cer = True  # Enable CER calculation
            model.wer.log_prediction = False  # Don't log predictions to save memory
            logging.info("Configured model WER metric to use CER")
        
        # Configure model logging for W&B
        if wandb_logger:
            # Enable detailed logging of training metrics
            model.log_train_metrics = True
            # Ensure validation metrics are logged
            if hasattr(model, 'log_validation_metrics'):
                model.log_validation_metrics = True
            logging.info("Enabled detailed training and validation metrics logging for W&B")
        
        # Calculate scheduler parameters based on dataset size
        if not skip_data_processing:
            train_dataset_size = len(train_data)
        else:
            # Estimate dataset size from manifest file for resume cases
            try:
                with open(train_manifest, 'r') as f:
                    train_dataset_size = sum(1 for line in f if line.strip())
            except:
                logging.warning("Could not determine dataset size from manifest, using default estimate")
                train_dataset_size = 10000  # Conservative estimate
        
        total_steps, warmup_steps, steps_per_epoch = calculate_scheduler_steps(
            dataset_size=train_dataset_size,
            batch_size=hparams['batch_size'],
            grad_accumulation=hparams['grad_accumulation_factor'],
            epochs=hparams['epochs'],
            debug_mode=hparams.get('debug_mode', False)
        )
        
        # Log scheduler information
        logging.info("=" * 50)
        logging.info("SCHEDULER CONFIGURATION")
        logging.info("=" * 50)
        logging.info(f"Dataset size: {train_dataset_size}")
        logging.info(f"Effective batch size: {hparams['batch_size'] * hparams['grad_accumulation_factor']}")
        logging.info(f"Steps per epoch: {steps_per_epoch}")
        logging.info(f"Total steps: {total_steps}")
        logging.info(f"Warmup steps: {warmup_steps}")
        logging.info(f"Learning rate: {hparams['learning_rate']}")
        logging.info(f"Min learning rate: {hparams['min_lr']}")
        logging.info("=" * 50)
        
        # Setup optimizer with NeMo's native pattern
        optimizer_conf = {
            'name': 'adamw',
            'lr': hparams["learning_rate"],
            'betas': [0.9, 0.98],
            'weight_decay': hparams["weight_decay"],
            'sched': {
                'name': hparams["scheduler_name"],
                'warmup_steps': warmup_steps,
                'min_lr': hparams["min_lr"],
                'max_steps': total_steps,
            }
        }
        
        model.setup_optimization(optimizer_conf)
        
        # Update W&B with scheduler information and additional metrics config
        if wandb_logger:
            try:
                # Calculate additional metrics for logging
                total_audio_hours = 0
                avg_audio_duration = 0
                if not skip_data_processing:
                    total_audio_hours = sum(item['duration'] for item in train_data) / 3600
                    avg_audio_duration = sum(item['duration'] for item in train_data) / len(train_data)
                else:
                    # Estimate from manifest if available
                    try:
                        with open(train_manifest, 'r') as f:
                            durations = []
                            for line in f:
                                if line.strip():
                                    data = json.loads(line)
                                    durations.append(data.get('duration', 0))
                            if durations:
                                total_audio_hours = sum(durations) / 3600
                                avg_audio_duration = sum(durations) / len(durations)
                    except:
                        pass
                
                wandb_logger.experiment.config.update({
                    'total_steps': total_steps,
                    'warmup_steps': warmup_steps,
                    'steps_per_epoch': steps_per_epoch,
                    'dataset_size': train_dataset_size,
                    'scheduler_info': f"CosineAnnealing with {warmup_steps} warmup steps",
                    # Additional dataset metrics
                    'total_audio_hours': round(total_audio_hours, 2),
                    'avg_audio_duration_sec': round(avg_audio_duration, 2),
                    'augmentation_factor': hparams['num_augmented_variants'] + 1 if hparams['augment'] else 1,
                    # Training efficiency metrics
                    'samples_per_second': round(train_dataset_size / (hparams['epochs'] * steps_per_epoch * hparams['grad_accumulation_factor']), 2),
                    'theoretical_training_time_hours': round((total_steps * hparams['grad_accumulation_factor']) / 3600, 2),
                })
                logging.info("Updated W&B config with scheduler and dataset information")
            except Exception as e:
                logging.warning(f"Failed to update W&B config: {e}")
        
        # Custom callback for additional metrics logging
        class AdditionalMetricsCallback(pl.Callback):
            def __init__(self):
                super().__init__()
                self.start_time = None
                
            def on_train_start(self, trainer, pl_module):
                import time
                self.start_time = time.time()
                if trainer.logger:
                    trainer.logger.log_metrics({
                        'system/gpu_count': trainer.num_devices,
                        'system/batch_size': hparams['batch_size'],
                        'system/effective_batch_size': hparams['batch_size'] * hparams['grad_accumulation_factor'],
                    })
            
            def on_train_epoch_end(self, trainer, pl_module):
                if trainer.logger and self.start_time:
                    import time
                    elapsed_time = time.time() - self.start_time
                    current_epoch = trainer.current_epoch + 1
                    
                    # Calculate training speed metrics
                    samples_processed = current_epoch * steps_per_epoch * hparams['batch_size'] * hparams['grad_accumulation_factor']
                    samples_per_second = samples_processed / elapsed_time if elapsed_time > 0 else 0
                    
                    trainer.logger.log_metrics({
                        'performance/samples_per_second': samples_per_second,
                        'performance/elapsed_time_hours': elapsed_time / 3600,
                        'performance/samples_processed': samples_processed,
                        'performance/epochs_per_hour': current_epoch / (elapsed_time / 3600) if elapsed_time > 0 else 0,
                    }, step=trainer.global_step)
            
            def on_validation_epoch_end(self, trainer, pl_module):
                if trainer.logger:
                    # Log additional validation metrics if available
                    try:
                        # Get current learning rate
                        current_lr = trainer.optimizers[0].param_groups[0]['lr']
                        trainer.logger.log_metrics({
                            'optimizer/learning_rate': current_lr,
                            'training/current_epoch': trainer.current_epoch,
                            'training/global_step': trainer.global_step,
                        }, step=trainer.global_step)
                        
                        # Log GPU memory usage if available
                        if torch.cuda.is_available():
                            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                            trainer.logger.log_metrics({
                                'system/gpu_memory_allocated_gb': gpu_memory_allocated,
                                'system/gpu_memory_reserved_gb': gpu_memory_reserved,
                            }, step=trainer.global_step)
                        
                        # Log sample predictions every few epochs
                        if hparams.get("log_predictions", False) and trainer.current_epoch % 2 == 0:
                            self._log_sample_predictions(trainer, pl_module)
                            
                    except Exception as e:
                        pass  # Don't fail training if metrics logging fails
            
            def _log_sample_predictions(self, trainer, pl_module):
                """Log sample predictions to W&B for qualitative analysis."""
                try:
                    # Get a few samples from validation set
                    val_dataloader = trainer.val_dataloaders[0] if trainer.val_dataloaders else None
                    if val_dataloader and hasattr(trainer.logger, 'experiment'):
                        import wandb
                        
                        # Get one batch for prediction logging
                        batch = next(iter(val_dataloader))
                        if len(batch) >= 3:  # audio, audio_len, targets
                            audio_signal, audio_lengths, targets = batch[:3]
                            
                            # Move to device
                            if torch.cuda.is_available():
                                audio_signal = audio_signal.cuda()
                                audio_lengths = audio_lengths.cuda()
                            
                            # Get predictions (first 3 samples only)
                            with torch.no_grad():
                                predictions = pl_module.transcribe(
                                    audio=audio_signal[:3], 
                                    audio_lengths=audio_lengths[:3]
                                )
                            
                            # Create prediction table
                            prediction_data = []
                            for i, pred in enumerate(predictions[:3]):
                                if i < len(targets):
                                    target_text = "N/A"  # Would need tokenizer to decode
                                    prediction_data.append([
                                        f"Sample_{trainer.current_epoch}_{i}",
                                        pred,
                                        target_text,
                                        len(pred.split()) if pred else 0
                                    ])
                            
                            # Log to W&B
                            table = wandb.Table(
                                columns=["Sample_ID", "Prediction", "Target", "Word_Count"],
                                data=prediction_data
                            )
                            trainer.logger.experiment.log({
                                f"predictions/epoch_{trainer.current_epoch}": table
                            }, step=trainer.global_step)
                            
                except Exception as e:
                    logging.warning(f"Failed to log sample predictions: {e}")
        
        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=hparams["save_folder"],
            filename='nemo-arabic-{epoch:02d}-{val_wer:.3f}',
            monitor='val_wer',
            mode='min',
            save_top_k=3,
            save_last=True,
        )
        
        early_stopping = EarlyStopping(
            monitor='val_wer',
            patience=3,
            mode='min',
        )
        
        # Add custom metrics callback
        additional_metrics_callback = AdditionalMetricsCallback()
        
        callbacks = [checkpoint_callback, early_stopping, additional_metrics_callback]
        trainer_config['callbacks'] = callbacks
        
        # Create trainer and set it on the model
        trainer = pl.Trainer(**trainer_config)
        model.set_trainer(trainer)
        
        # Final validation before training
        if warmup_steps >= total_steps:
            logging.warning(f"Warmup steps ({warmup_steps}) >= total steps ({total_steps}). Adjusting warmup.")
            warmup_steps = max(1, total_steps // 4)  # Use 25% as fallback
            logging.info(f"Adjusted warmup steps to: {warmup_steps}")
        
        logging.info("Starting training...")
        logging.info(f"Final scheduler config: warmup={warmup_steps}, total={total_steps}, lr={hparams['learning_rate']}")
        if resume_checkpoint_path:
            logging.info(f"Resuming training from checkpoint: {resume_checkpoint_path}")
        else:
            logging.info("Starting fresh training")
        
        # Use the model's trainer instead of calling trainer.fit directly
        try:
            model.trainer.fit(model)
        except Exception as e:
            if resume_checkpoint_path:
                logging.error(f"Training failed with resume checkpoint: {e}")
                logging.info("You may want to try force_restart=True to start fresh")
            raise
        
        # Test model
        logging.info("Running final evaluation...")
        trainer.test(model, ckpt_path='best')
        
        # Save final model
        final_model_path = os.path.join(hparams["save_folder"], "final_model.nemo")
        model.save_to(final_model_path)
        logging.info(f"Final model saved to: {final_model_path}")
        
        # Commit volume
        volume.commit()
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        volume.commit()  # Save any partial progress
        raise
    
    finally:
        # W&B cleanup is handled automatically by PyTorch Lightning WandbLogger
        # but we can explicitly finalize if needed
        if wandb_logger:
            try:
                wandb_logger.finalize("success")
                logging.info("W&B logging finalized successfully")
            except Exception as e:
                logging.warning(f"W&B finalization warning: {e}")

@app.local_entrypoint()
def main():
    """Local entrypoint to start training."""
    # This check prevents the script from being re-executed inside the container
    # when `modal run` is called.
    if sys.flags.interactive:
        return
    print("Starting NeMo Arabic ASR fine-tuning...")
    train_nemo_arabic.remote()
    print("Training job submitted to Modal.") 