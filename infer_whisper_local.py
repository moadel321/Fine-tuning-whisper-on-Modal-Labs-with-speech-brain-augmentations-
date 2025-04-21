import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import argparse
import os
import logging
# Imports for Arabic text display
import arabic_reshaper
from bidi.algorithm import get_display
# Imports for API calls
from openai import OpenAI
import azure.cognitiveservices.speech as speechsdk
import time # For Azure wait loop

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_audio(audio_path, target_sr=16000):
    """Loads and preprocesses audio for Whisper."""
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            logging.info(f"Resampling audio from {sr} Hz to {target_sr} Hz")
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)

        # Ensure mono
        if waveform.shape[0] > 1:
            logging.info("Converting audio to mono")
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ensure float32
        waveform = waveform.float()

        logging.info(f"Loaded audio: {audio_path}, Duration: {waveform.shape[1]/target_sr:.2f}s")
        return waveform.squeeze(0) # Remove batch dim for processor

    except Exception as e:
        logging.error(f"Error loading or processing audio file {audio_path}: {e}", exc_info=True)
        return None

def load_model_from_speechbrain_ckpt(model_hub, ckpt_path, device):
    """Loads a Hugging Face Whisper model from a SpeechBrain checkpoint."""
    logging.info(f"Loading base model and processor from: {model_hub}")
    try:
        processor = WhisperProcessor.from_pretrained(model_hub)
        model = WhisperForConditionalGeneration.from_pretrained(model_hub)
    except Exception as e:
        logging.error(f"Failed to load base model/processor from {model_hub}: {e}", exc_info=True)
        return None, None

    logging.info(f"Loading SpeechBrain checkpoint from: {ckpt_path}")
    try:
        # Load the checkpoint file
        # map_location avoids loading GPU tensors if you only have CPU
        checkpoint = torch.load(ckpt_path, map_location=device)

        # --- Determine the correct state dictionary ---
        sb_state_dict = None
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                 sb_state_dict = checkpoint['model']
                 logging.info("Found state dict under 'model' key.")
            elif 'whisper' in checkpoint:
                 if hasattr(checkpoint['whisper'], 'state_dict'):
                      sb_state_dict = checkpoint['whisper'].state_dict() # If it's the module object itself
                      logging.info("Found state dict under 'whisper' key (module object).")
                 elif isinstance(checkpoint['whisper'], dict):
                      sb_state_dict = checkpoint['whisper'] # If it's already the state dict
                      logging.info("Found state dict under 'whisper' key (dictionary).")
            elif checkpoint: # Check if the checkpoint dict itself is not empty
                 # Assume the checkpoint itself is the state dict if keys look like model params
                 # (A more robust check could be added here if needed)
                 logging.info("Keys 'model' or 'whisper' not found. Assuming the checkpoint file directly contains the state dict.")
                 sb_state_dict = checkpoint
            else:
                 logging.error("Checkpoint dictionary is empty or invalid after loading.")
                 return None, None
        else:
             # If checkpoint is not a dict, maybe it's the raw state_dict saved directly?
             # This case is less common with SpeechBrain Checkpointer but possible.
             logging.warning("Checkpoint loaded is not a dictionary. Attempting to use it directly as state_dict.")
             sb_state_dict = checkpoint # Treat the loaded object itself as the state_dict

        if sb_state_dict is None:
             logging.error("Could not extract a valid state dictionary from the checkpoint.")
             return None, None

        # --- Load the state dictionary into the Hugging Face model ---
        logging.info(f"Attempting to load state dict into {model.__class__.__name__}...")
        try:
            # Ensure sb_state_dict is actually a dictionary before proceeding
            if not isinstance(sb_state_dict, dict):
                 logging.error(f"Object intended as state_dict is not a dictionary (type: {type(sb_state_dict)}). Cannot load.")
                 return None, None

            # Check for 'model.' prefix (common in SB checkpoints) and remove if present
            has_model_prefix = any(k.startswith("model.") for k in sb_state_dict.keys())
            if has_model_prefix:
                logging.info("Detected 'model.' prefix in checkpoint keys. Removing prefix...")
                sb_state_dict = {k.removeprefix("model."): v for k, v in sb_state_dict.items() if k.startswith("model.")}
                # Add a check if filtering resulted in an empty dict
                if not sb_state_dict:
                     logging.error("State dictionary became empty after removing 'model.' prefix. Checkpoint structure might be unexpected.")
                     return None, None

            missing_keys, unexpected_keys = model.load_state_dict(sb_state_dict, strict=False)

            if unexpected_keys:
                 logging.warning(f"Unexpected keys found in checkpoint state dict: {unexpected_keys}")
                 # Filter out unexpected keys if they cause issues (e.g., from optimizer state)
                 # You might need to inspect these keys further.
                 # Example: sb_state_dict = {k: v for k, v in sb_state_dict.items() if k in model.state_dict()}
                 # model.load_state_dict(sb_state_dict, strict=True) # Try again with strict=True if filtered

            if missing_keys:
                 logging.warning(f"Missing keys in model state dict (might be ok if they are non-essential, like final_logits_bias): {missing_keys}")

            logging.info("Successfully loaded state dict into model.")

        except RuntimeError as e:
             logging.error(f"RuntimeError loading state dict: {e}")
             logging.info("This often indicates a mismatch between the checkpoint and the model architecture.")
             logging.info("Ensure `model_hub` matches the base model used for fine-tuning.")
             return None, None
        except Exception as e:
             logging.error(f"Error loading state dict: {e}", exc_info=True)
             return None, None

        model.to(device)
        model.eval() # Set to evaluation mode
        logging.info(f"Model loaded successfully and moved to {device}.")
        return processor, model

    except FileNotFoundError:
        logging.error(f"Checkpoint file not found: {ckpt_path}")
        return None, None
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}", exc_info=True)
        return None, None

def transcribe_local(processor, model, audio_waveform, language, task, device, num_beams=5):
    """Transcribes audio using the loaded local Whisper model."""
    if processor is None or model is None or audio_waveform is None:
        logging.error("Processor, model, or audio waveform is None. Cannot transcribe locally.")
        return "Local transcription failed."

    logging.info(f"Processing audio for local model input...")
    try:
        # Ensure processor language/task is set correctly (redundant if loaded right, but safe)
        processor.tokenizer.set_prefix_tokens(language=language, task=task)

        # The processor expects a raw waveform numpy array or list of floats
        input_features = processor(audio_waveform.numpy(), sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(device)
    except Exception as e:
        logging.error(f"Error during local feature extraction: {e}", exc_info=True)
        return "Local feature extraction failed."

    # --- Generate token IDs ---
    # Prepare the decoder input IDs for generation (SOT, lang, task, no_timestamps)
    # We get the specific IDs from the *processor's tokenizer*
    decoder_start_token_id = model.config.decoder_start_token_id # Usually SOT <|startoftranscript|>
    lang_token_id = processor.tokenizer.convert_tokens_to_ids(f"<|{language}|>")
    task_token_id = processor.tokenizer.convert_tokens_to_ids(f"<|{task}|>")
    no_timestamps_token_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

    # Check if tokens were found
    if any(t is None for t in [lang_token_id, task_token_id, no_timestamps_token_id]):
         logging.error(f"Could not get all required token IDs for local model: lang={lang_token_id}, task={task_token_id}, notimestamps={no_timestamps_token_id}")
         # Fallback: Use default forced_decoder_ids if necessary, but might be wrong lang/task
         forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
         logging.warning(f"Using processor.get_decoder_prompt_ids as fallback for local model: {forced_decoder_ids}")
    else:
         # Standard forced IDs for transcription
         forced_decoder_ids = [
             (1, lang_token_id),
             (2, task_token_id),
             (3, no_timestamps_token_id),
         ]
         logging.info(f"Using forced_decoder_ids for local model: {forced_decoder_ids}")


    logging.info(f"Generating local transcription (num_beams={num_beams})...")
    try:
        with torch.no_grad(): # Disable gradient calculations for inference
            # Use the model's generate method
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=model.config.max_length, # Use model's default max length
                num_beams=num_beams,
                # early_stopping=True # Optional: Stop generation earlier
            )
    except Exception as e:
        logging.error(f"Error during local model generation: {e}", exc_info=True)
        return "Local model generation failed."

    logging.info("Decoding generated token IDs for local model...")
    try:
        # Decode the generated IDs to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        # batch_decode returns a list, get the first element
        return transcription[0] if transcription else ""
    except Exception as e:
        logging.error(f"Error during local token decoding: {e}", exc_info=True)
        return "Local token decoding failed."

def transcribe_openai(api_key, audio_path, language):
    """Transcribes audio using the OpenAI API."""
    if not api_key:
        logging.warning("OpenAI API key not provided. Skipping OpenAI transcription.")
        return None

    logging.info(f"Transcribing with OpenAI API (language={language})...")
    try:
        client = OpenAI(api_key=api_key)
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                language=language # Specify the language
            )
        logging.info("OpenAI transcription successful.")
        # The response object has a 'text' attribute
        return transcript.text
    except Exception as e:
        logging.error(f"Error during OpenAI transcription: {e}", exc_info=True)
        return f"OpenAI transcription failed: {e}"

def transcribe_azure(speech_key, service_region, audio_path, language_locale="ar-EG"):
    """Transcribes audio using the Azure Speech Service API."""
    if not speech_key or not service_region:
        logging.warning("Azure Speech key or region not provided. Skipping Azure transcription.")
        return None

    logging.info(f"Transcribing with Azure API (locale={language_locale})...")
    try:
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        speech_config.speech_recognition_language = language_locale

        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        logging.info("Starting Azure single-shot recognition...")
        result = speech_recognizer.recognize_once_async().get() # Use recognize_once for simplicity with files

        # Check the result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            logging.info("Azure transcription successful.")
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            logging.error(f"Azure NoMatch: Speech could not be recognized. Details: {result.no_match_details}")
            return "Azure transcription failed: NoMatch"
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logging.error(f"Azure Canceled: Reason={cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                logging.error(f"Azure ErrorDetails: {cancellation_details.error_details}")
            return f"Azure transcription failed: Canceled ({cancellation_details.reason})"
        else:
            logging.error(f"Azure transcription failed with unknown reason: {result.reason}")
            return f"Azure transcription failed: Unknown reason ({result.reason})"

    except Exception as e:
        logging.error(f"Error during Azure transcription: {e}", exc_info=True)
        return f"Azure transcription failed: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Whisper inference locally and optionally compare with APIs.")
    # Local model args
    parser.add_argument("--model_hub", type=str, default="openai/whisper-small", help="Base Whisper model hub path (e.g., 'openai/whisper-small'). MUST match fine-tuning base.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the SpeechBrain model.ckpt file.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file to transcribe.")
    parser.add_argument("--language", type=str, default="ar", help="Language token for local model and OpenAI API (e.g., 'ar', 'en').")
    parser.add_argument("--task", type=str, default="transcribe", help="Task token for local model ('transcribe' or 'translate').")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for local model inference ('cuda' or 'cpu').")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for local model beam search decoding.")
    # OpenAI args
    parser.add_argument("--openai_api_key", type=str, default=None, help="Your OpenAI API key.")
    # Azure args
    parser.add_argument("--azure_speech_key", type=str, default=None, help="Your Azure Speech resource key.")
    parser.add_argument("--azure_service_region", type=str, default=None, help="Your Azure Speech service region (e.g., 'westeurope').")
    parser.add_argument("--azure_language_locale", type=str, default="ar-EG", help="Azure language locale (e.g., 'ar-EG', 'ar-SA', 'en-US').")
    # Output args
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save transcription text files.")

    args = parser.parse_args()

    # --- Basic File Checks ---
    if not os.path.exists(args.ckpt_path):
        logging.error(f"Checkpoint file not found: {args.ckpt_path}")
        exit(1)
    if not os.path.exists(args.audio_path):
        logging.error(f"Audio file not found: {args.audio_path}")
        exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Setup --- 
    device = torch.device(args.device)
    logging.info(f"Using device for local model: {device}")
    results = {}
    base_filename = os.path.splitext(os.path.basename(args.audio_path))[0]

    # --- Load Local Model --- 
    processor, model = load_model_from_speechbrain_ckpt(args.model_hub, args.ckpt_path, device)

    # --- Load Audio --- 
    # Load audio only once for all models
    audio_waveform = load_audio(args.audio_path)
    if audio_waveform is None:
        print("Failed to process audio file. Exiting.")
        exit(1)

    # --- Run Local Transcription --- 
    if processor and model:
        results['local'] = transcribe_local(
            processor, model, audio_waveform,
            args.language, args.task, device, args.num_beams
        )
    else:
        results['local'] = "Failed to load local model or processor."
        print(results['local'])

    # --- Run OpenAI Transcription (Conditional) ---
    if args.openai_api_key:
        results['openai'] = transcribe_openai(args.openai_api_key, args.audio_path, args.language)
    else:
        logging.info("Skipping OpenAI transcription (no API key provided).")

    # --- Run Azure Transcription (Conditional) ---
    if args.azure_speech_key and args.azure_service_region:
        results['azure'] = transcribe_azure(
            args.azure_speech_key, args.azure_service_region,
            args.audio_path, args.azure_language_locale
        )
    else:
        logging.info("Skipping Azure transcription (no key or region provided).")

    # --- Print Results and Save to Files ---
    print("\n" + "=" * 15 + " TRANSCRIPTION RESULTS " + "=" * 15)
    print(f"Audio File: {args.audio_path}")

    for model_name, transcription in results.items():
        print("-" * 40)
        # Handle potential None values if API calls were skipped
        transcription_text = transcription if transcription is not None else "[SKIPPED]"
        # Display processing for Arabic
        if args.language == "ar" and transcription_text != "[SKIPPED]" and not "failed" in transcription_text.lower():
            try:
                 reshaped_text = arabic_reshaper.reshape(transcription_text)
                 bidi_text = get_display(reshaped_text)
                 print(f"{model_name.upper()} Transcription:\n{bidi_text}")
            except Exception as display_e:
                 logging.warning(f"Could not process Arabic display for {model_name}: {display_e}")
                 print(f"{model_name.upper()} Transcription (raw):\n{transcription_text}") # Print raw if display fails
        else:
             print(f"{model_name.upper()} Transcription:\n{transcription_text}")

        # Save raw transcription to file
        output_filename = os.path.join(args.output_dir, f"{base_filename}__{model_name}.txt")
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(transcription_text)
            logging.info(f"Saved {model_name} transcription to: {output_filename}")
        except Exception as save_e:
            logging.error(f"Failed to save {model_name} transcription to {output_filename}: {save_e}")

    print("=" * 51) 