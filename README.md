# Whisper Fine-Tuning and Inference on Modal

This project provides scripts to fine-tune an OpenAI Whisper model on a custom dataset using [Modal](https://modal.com/) for GPU-accelerated training and then run inference locally using the fine-tuned checkpoint, with optional comparison against OpenAI and Azure transcription APIs.

## Project Goals

*   Fine-tune Whisper (specifically `openai/whisper-small` by default) on a custom Arabic dataset (e.g., Egyptian dialect).
*   Leverage Modal for scalable, serverless GPU training without managing infrastructure.
*   Provide a script for easy local inference using the fine-tuned model.
*   Offer comparison with commercial transcription APIs (OpenAI, Azure) for benchmarking.

## Code Structure

*   `train_whisper_modal.py`: Script to define and run the fine-tuning process on Modal.
*   `infer_whisper_local.py`: Script to load a fine-tuned checkpoint and perform transcription locally or compare with APIs.
*   `.env`: (To be created by user) Stores API keys securely.
*   `README.md`: This file.

## Setup

### Prerequisites

1.  **Python:** Python 3.10 or later installed locally.
2.  **Modal Account:** Sign up for a free Modal account at [modal.com](https://modal.com/).
3.  **Modal CLI:** Install and set up the Modal CLI:
    ```bash
    pip install modal-client
    modal setup
    ```
4.  **Git:** Clone this repository:
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
5.  **Hugging Face Account & Token:**
    *   You need a Hugging Face account ([huggingface.co](https://huggingface.co/)).
    *   Create a User Access Token with **write** permissions on the [Tokens settings page](https://huggingface.co/settings/tokens). This is needed if the script pushes the model (though current scripts focus on saving to Modal Volume).
6.  **(Optional) Weights & Biases Account:** If you want to use W&B logging, create an account at [wandb.ai](https://wandb.ai/).
7.  **(Optional) API Keys:** If using `infer_whisper_local.py` to compare with APIs:
    *   **OpenAI API Key:** Get from [platform.openai.com](https://platform.openai.com/).
    *   **Azure Speech Service Key & Region:** Create a Speech resource in the Azure portal.

### Modal Secrets

Modal uses Secrets to securely store credentials. Create the following secrets via the [Modal Secrets page](https://modal.com/secrets) or CLI:

1.  **Hugging Face Token:** Store your HF write token.
    ```bash
    modal secret create huggingface-secret-write HF_TOKEN=<your-huggingface-write-token>
    ```
2.  **Weights & Biases API Key (Optional):** If using W&B.
    ```bash
    modal secret create wandb-secret WANDB_API_KEY=<your-wandb-api-key>
    ```

### Environment Variables (`.env` file)

Create a file named `.env` in the root of the project directory to store API keys for local inference (if needed). **Do not commit this file to Git.**

```dotenv
# .env file
OPENAI_API_KEY="sk-..."
AZURE_SPEECH_KEY="..."
AZURE_SERVICE_REGION="YourAzureRegion" # e.g., westeurope, eastus
```

`infer_whisper_local.py` will automatically load these variables.

## Training (`train_whisper_modal.py`)

This script defines a Modal function to perform the fine-tuning process on a GPU instance in the cloud.

### Configuration

Before running, review and modify the placeholders and `hparams` dictionary within `train_whisper_modal.py`:

*   **Modal App Name & Volume Name:** Search for `Placeholder` comments near the `modal.App` and `modal.Volume.from_name` definitions and replace them with your desired names.
*   **Dataset:**
    *   `hparams["hf_dataset_id"]`: Change `"huggingfaceusername/datasetname"` to the Hugging Face dataset identifier for your custom dataset (e.g., `"MAdel121/arabic-egy-cleaned"`). Ensure the dataset has 'audio' and 'text' columns. The script assumes standard splits like 'train', 'validation', 'test'.
*   **Base Model:**
    *   `hparams["whisper_hub"]`: Change `"openai/whisper-small"` if you want to fine-tune a different Whisper variant (e.g., `openai/whisper-base`, `openai/whisper-medium`). **Ensure this matches the base model used in `infer_whisper_local.py`**.
*   **Language & Task:**
    *   `hparams["language"]`: Set the target language code (e.g., `"ar"`, `"en"`).
    *   `hparams["task"]`: Usually `"transcribe"`.
*   **Paths:**
    *   `hparams["save_folder"]`, `hparams["output_folder"]`: Define paths within the Modal volume where checkpoints and logs will be saved. Usually, the defaults are fine.
*   **Augmentation:**
    *   `hparams["augment"]`: `True` or `False` to enable/disable augmentation.
    *   `hparams["augment_prob_master"]`: Overall probability of applying augmentation per batch.
    *   `hparams["use_*"]` flags: Toggle specific augmentations (`AddNoise`, `AddReverb`, `SpeedPerturb`, etc.).
    *   Modify parameters for enabled augmentations (SNR levels, drop lengths, etc.) as needed.
*   **Training Parameters:**
    *   `hparams["epochs"]`, `hparams["learning_rate"]`, `hparams["loader_batch_size"]`, `hparams["grad_accumulation_factor"]`, etc.: Adjust these based on your dataset size, GPU memory, and desired training regime.
*   **Weights & Biases (Optional):**
    *   `hparams["use_wandb"]`: Set to `True` to enable logging.
    *   `hparams["wandb_project"]`: Change `"you project's name on weights and biases "` to your project name.
    *   `hparams["wandb_entity"]`: Optionally set your W&B username or team name.
    *   `hparams["wandb_resume_id"]`: Set to a specific run ID string (e.g., `"ceeu3g6c"`) to resume a previous W&B run. Leave as `None` for a new run.
*   **GPU Type:**
    *   Modify the `gpu="A100-40GB"` argument in the `@app.function` decorator if you need a different GPU type available on Modal (e.g., `"T4"`, `"A10G"`).

### Running the Training

Execute the script locally using the Modal CLI:

```bash
modal run train_whisper_modal.py
```

This command deploys the code to Modal and starts the `train_whisper_on_modal` function on a remote container with the specified GPU.

*   Monitor the progress via the logs printed in your terminal and the Modal dashboard.
*   If W&B is enabled, track the run via your W&B project page.
*   Checkpoints will be saved periodically to the persistent Modal Volume you named during setup.

## Inference (`infer_whisper_local.py`)

This script loads a fine-tuned SpeechBrain checkpoint (saved to the Modal Volume during training) and performs transcription on a local audio file. It can also optionally transcribe the same audio using OpenAI and Azure APIs for comparison.

### Downloading the Checkpoint

Before running inference, you need the fine-tuned checkpoint file (`model.ckpt`) saved during training. Download it from your Modal Volume:

1.  Find the path to your checkpoint within the volume (e.g., `/root/checkpoints/whisper_small_egy_save/CKPT+.../model.ckpt`).
2.  Use the Modal CLI or Python client to download the file. Example using CLI (requires `modal volume get` command knowledge - refer to Modal docs):
    ```bash
    # Example - syntax might vary, check Modal docs for volume file operations
    # modal volume get <your-volume-name> /root/checkpoints/whisper_small_egy_save/CKPT+.../model.ckpt ./downloaded_model.ckpt
    ```
    Alternatively, add a simple Modal function to list files or download specific ones.

### Running Inference

Execute the script from your terminal:

```bash
python infer_whisper_local.py --ckpt_path /path/to/your/downloaded_model.ckpt --audio_path /path/to/your/audio.wav [OPTIONS]
```

**Required Arguments:**

*   `--ckpt_path`: Path to the downloaded `.ckpt` file from your fine-tuning run.
*   `--audio_path`: Path to the audio file (e.g., `.wav`, `.mp3`) you want to transcribe.

**Optional Arguments:**

*   `--model_hub`: Base Whisper model used for fine-tuning (e.g., `"openai/whisper-small"`). **Crucially, this MUST match the `whisper_hub` used during training.** Defaults to `"openai/whisper-small"`.
*   `--language`: Language code (e.g., `"ar"`, `"en"`). Used for local model decoding and OpenAI API. Defaults to `"ar"`.
*   `--task`: Task for the local model (`"transcribe"` or `"translate"`). Defaults to `"transcribe"`.
*   `--device`: Device for inference (`"cuda"` or `"cpu"`). Defaults to CUDA if available.
*   `--num_beams`: Number of beams for local model generation. Defaults to `5`.
*   `--openai_api_key`: OpenAI API key. If not provided, uses `OPENAI_API_KEY` from `.env`. Skips OpenAI if neither is found.
*   `--azure_speech_key`: Azure Speech key. If not provided, uses `AZURE_SPEECH_KEY` from `.env`. Skips Azure if not found.
*   `--azure_service_region`: Azure Speech region. If not provided, uses `AZURE_SERVICE_REGION` from `.env`. Skips Azure if not found.
*   `--azure_language_locale`: Specific locale for Azure (e.g., `"ar-EG"`, `"en-US"`). Defaults to `"ar-EG"`.
*   `--output_dir`: Directory to save the transcription results as text files. Defaults to the current directory (`.`).

### Output

*   Transcriptions from the local model (and APIs, if configured) will be printed to the console. Arabic text will be reshaped for proper display.
*   Separate text files for each model's transcription (e.g., `audio__local.txt`, `audio__openai.txt`) will be saved in the specified `--output_dir`.

## Customization

*   **Language:** Change the `language` parameter in `hparams` (`train_whisper_modal.py`) and the `--language` / `--azure_language_locale` arguments (`infer_whisper_local.py`).
*   **Dataset:** Update `hparams["hf_dataset_id"]` to point to your Hugging Face dataset. Ensure it's compatible (audio, text columns).
*   **Whisper Model:** Change `hparams["whisper_hub"]` and the `--model_hub` argument consistently to use different Whisper model sizes. Note that larger models require more GPU memory.
*   **Augmentations:** Experiment with enabling/disabling different augmentations and tuning their parameters in `hparams` to potentially improve model robustness.

## Troubleshooting

*   **Modal Errors:** Ensure you have the latest Modal client (`pip install --upgrade modal-client`), are logged in (`modal token set`), and have correctly created the necessary secrets. Check Modal dashboard logs for detailed error messages from the container.
*   **Checkpoint Loading Errors (`infer_whisper_local.py`):**
    *   Make sure the `--model_hub` argument exactly matches the base model used for training (`hparams["whisper_hub"]`). Architecture mismatches are common errors.
    *   Verify the `--ckpt_path` points to the correct, fully downloaded `model.ckpt` file.
*   **API Errors (`infer_whisper_local.py`):**
    *   Ensure API keys and regions/locales are correct in your `.env` file or command-line arguments.
    *   Check network connectivity.
    *   Consult OpenAI/Azure documentation for specific API error codes.
*   **CUDA Errors:** Ensure PyTorch with CUDA support is installed correctly locally if using `--device cuda` for inference. Check GPU driver compatibility.
*   **W&B Errors:** Ensure the API key is correct in the Modal secret and that the project/entity names in `hparams` are valid.
*   **File Not Found:** Double-check all file paths (`--audio_path`, `--ckpt_path`).
