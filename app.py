import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import gradio as gr

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the Whisper model and processor for transcription
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
processor.feature_extractor.sampling_rate = 16000  # Ensure the sampling rate is set correctly

# Set up the transcription pipeline
transcription_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Function to transcribe and process text
def transcribe_and_process(audio):
    if audio is None:
        return "Please upload an audio file.", ""

    try:
        # Transcription
        result = transcription_pipe(audio)
        text = result["text"]

        # Further processing
        result2 = transcription_pipe(audio, generate_kwargs={"language": "english"})
        text2 = result2["text"]

        return text, text2
    except Exception as e:
        return f"An error occurred: {e}", ""

# Gradio interface setup
inputs = gr.Audio(type="filepath", label="Upload Audio File")
outputs = [
    gr.Textbox(label="Transcription"),
    gr.Textbox(label="Processed Text")
]

examples = [
    ["ben.mp3"] 
]

app = gr.Interface(
    fn=transcribe_and_process,
    inputs=inputs,
    outputs=outputs,
    title="Audio Transcription and Processing",
    examples=examples,
)
app.launch(share=True)
