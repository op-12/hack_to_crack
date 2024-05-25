# Speech Translation App: `https://huggingface.co/spaces/diaL42/speechTranscription`

## Note
**NOTE:** The deployed website `https://huggingface.co/spaces/diaL42/speechTranscription` uses CPU for execution; therefore, the output might be a bit slow. For optimal performance, it's recommended to use the .ipynb file mentioned above using colab GPU.

## Overview
This Speech Translation App is powered by Whisper v3 and Gradio, providing a seamless experience for translating spoken language into text. It leverages various libraries such as Torch, Transformers, Datasets, Accelerate, IPython, Gradio, and Sentencepiece to offer accurate and efficient translation capabilities.

## Requirements
- Torch
- Transformers
- Datasets
- Accelerate
- IPython
- Gradio
- Sentencepiece

## Automatic Speech Recognition (ASR) with Whisper:
```
* Model: openai/whisper-large-v3

* Description:
Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. Trained on 680k hours of labelled data, Whisper models demonstrate a strong ability to generalise to many datasets and domains without the need for fine-tuning.

Whisper was proposed in the paper Robust Speech Recognition via Large-Scale Weak Supervision by Alec Radford et al. from OpenAI. The original code repository can be found here.

Whisper large-v3 has the same architecture as the previous large models except the following minor differences:

The input uses 128 Mel frequency bins instead of 80
A new language token for Cantonese
The Whisper large-v3 model is trained on 1 million hours of weakly labeled audio and 4 million hours of pseudolabeled audio collected using Whisper large-v2. The model was trained for 2.0 epochs over this mixture dataset.

The large-v3 model shows improved performance over a wide variety of languages, showing 10% to 20% reduction of errors compared to Whisper large-v2.

Model details
Whisper is a Transformer based encoder-decoder model, also referred to as a sequence-to-sequence model. It was trained on 1 million hours of weakly labeled audio and 4 million hours of pseudolabeled audio collected using Whisper large-v2.

The models were trained on either English-only data or multilingual data. The English-only models were trained on the task of speech recognition. The multilingual models were trained on both speech recognition and speech translation. For speech recognition, the model predicts transcriptions in the same language as the audio. For speech translation, the model predicts transcriptions to a different language to the audio.

Whisper checkpoints come in five configurations of varying model sizes. The smallest four are trained on either English-only or multilingual data. The largest checkpoints are multilingual only. All ten of the pre-trained checkpoints are available on the Hugging Face Hub. The checkpoints are summarised in the following table with links to the models on the Hub:

* Functionality: Whisper processes audio inputs and generates corresponding transcriptions. It is capable of handling diverse linguistic inputs and produces accurate text representations of the spoken words.

* Pipeline Setup: The ASR pipeline integrates the Whisper model with tokenizers and feature extractors, ensuring the input audio is appropriately preprocessed and the output text is accurately tokenized.

* Usage: The model processes uploaded audio files, transcribing the speech into English text. Parameters such as max_new_tokens, chunk_length_s, and batch_size are configured to balance performance and accuracy.

```
## Installation
You can install the required dependencies using pip:
```bash
pip install torch transformers datasets accelerate ipython gradio sentencepiece



