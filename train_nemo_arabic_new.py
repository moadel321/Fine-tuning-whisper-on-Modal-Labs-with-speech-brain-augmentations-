import logging
import datetime
import json
import os
import sys
import random
import torch
import torchaudio
import soundfile as sf

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
    "debug_mode": True, # Flag for test runs on a small data subset
    "debug_num_samples": 1000, # Number of samples for debug mode
    "seed": 1986,
    "epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "batch_size": 8,
    "num_workers": 8,
    "grad_accumulation_factor": 4,
    "max_grad_norm": 5.0,
    
    # W&B
    "use_wandb": True,
    "wandb_project": "nemo-arabic-asr",
    "wandb_entity": None,
}

print("Hyperparameters defined.")

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

@app.function(
    image=modal_image,
    gpu="A100-80GB",
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
    
    # Initialize W&B
    wandb_run = None
    if hparams["use_wandb"]:
        try:
            import wandb
            wandb_run = wandb.init(
                project=hparams["wandb_project"],
                entity=hparams["wandb_entity"],
                config=hparams,
                name=f"nemo-arabic-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )
            logging.info(f"W&B initialized: {wandb_run.url}")
        except Exception as e:
            logging.warning(f"W&B initialization failed: {e}")
            hparams["use_wandb"] = False
    
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
        manifest_dir = os.path.join(hparams["output_folder"], "manifests")
        os.makedirs(manifest_dir, exist_ok=True)
        
        # Process training data
        logging.info("Processing training data...")
        train_data = process_split(
            raw_datasets[hparams["train_split"]],
            train_audio_dir,
            augment_fn=augment_and_save if hparams["augment"] and augmenter else None
        )
        train_manifest = os.path.join(manifest_dir, "train_manifest.json")
        create_nemo_manifest(train_data, train_manifest)
        
        # Process validation data (no augmentation)
        logging.info("Processing validation data...")
        val_data = process_split(
            raw_datasets[hparams["valid_split"]],
            val_audio_dir
        )
        val_manifest = os.path.join(manifest_dir, "val_manifest.json")
        create_nemo_manifest(val_data, val_manifest)
        
        # Process test data (no augmentation)
        logging.info("Processing test data...")
        test_data = process_split(
            raw_datasets[hparams["test_split"]],
            test_audio_dir
        )
        test_manifest = os.path.join(manifest_dir, "test_manifest.json")
        create_nemo_manifest(test_data, test_manifest)
        
        # Load NeMo model
        logging.info("Loading NeMo model...")
        model = EncDecHybridRNNTCTCBPEModel.from_pretrained(hparams["nemo_model_name"])
        
        # Configure model for Arabic ASR and update paths
        with open_dict(model.cfg):
            model.cfg.use_cer = True
            model.cfg.tokenizer.dir = hparams["save_folder"]
            # Update dataset configs to prevent warnings
            model.cfg.train_ds.manifest_filepath = train_manifest
            model.cfg.train_ds.is_tarred = False
            model.cfg.train_ds.batch_size = hparams['batch_size']
            model.cfg.validation_ds.manifest_filepath = val_manifest
            model.cfg.validation_ds.batch_size = hparams['batch_size']
            model.cfg.test_ds.manifest_filepath = test_manifest
            model.cfg.test_ds.batch_size = hparams['batch_size']
        
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
            'val_check_interval': 0.5,
            'enable_checkpointing': True,
            'logger': False,
            'enable_progress_bar': True,
        }
        
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
        
        # Set model to training mode
        model.train()
        
        # Additional model configuration for evaluation
        model.log_predictions = False
        model.compute_eval_loss = False
        
        # Setup optimizer (following NeMo pattern)
        optimizer_conf = {}
        optimizer_conf['name'] = 'adamw'
        optimizer_conf['lr'] = hparams["learning_rate"]
        optimizer_conf['betas'] = [0.9, 0.98]
        optimizer_conf['weight_decay'] = hparams["weight_decay"]
        
        # Setup scheduler
        sched = {}
        sched['name'] = 'CosineAnnealing'
        sched['warmup_steps'] = None
        sched['warmup_ratio'] = 0.10
        sched['min_lr'] = 1e-6
        optimizer_conf['sched'] = sched
        
        model.setup_optimization(optimizer_conf)
        
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
        
        callbacks = [checkpoint_callback, early_stopping]
        trainer_config['callbacks'] = callbacks
        
        # Create trainer and set it on the model
        trainer = pl.Trainer(**trainer_config)
        model.set_trainer(trainer)
        
        logging.info("Starting training...")
        # Use the model's trainer instead of calling trainer.fit directly
        model.trainer.fit(model)
        
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
        if wandb_run:
            wandb.finish()

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