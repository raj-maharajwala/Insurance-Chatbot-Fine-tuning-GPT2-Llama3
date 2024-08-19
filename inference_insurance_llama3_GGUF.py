import os
import psutil
import time
from pathlib import Path
from llama_cpp import Llama
from textwrap import dedent
import glob
import multiprocessing
from huggingface_hub import hf_hub_download

def download_model():
    model_name = "Raj-Maharajwala/OpenInsuranceLLM-Llama3-8B-GGUF"
    model_file = "openinsurancellm-llama3-8b.Q5_K_M.gguf"
    model_1_path = hf_hub_download(model_name,
                                    filename = model_file,
                                    local_dir = os.path.join(os.getcwd(), 'gguf_dir'))
    return model_1_path

def load_model(n_ctx, n_threads, n_batch, n_gpu_layers):
    quantized_path = "gguf_dir/" 
    MODEL_DIR = os.path.join(os.getcwd(), quantized_path)
    try:
      directory = Path(MODEL_DIR)
      model_1_path = str(list(directory.glob('openinsurancellm*Q5*.gguf'))[0])
    except:
      model_1_path = download_model()


    llm_ctx = Llama(model_path=model_1_path,
                    n_gpu_layers=n_gpu_layers,  # No GPU layers
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_batch=n_batch,
                    verbose=False)
    return llm_ctx


def get_prompt(Question):
    System = """You are an expert and experienced from the Insurance domain with extensive insurance knowledge and professional writter with all the insurance policies.
    Your name is OpenInsuranceLLM, and you were developed by Raj Maharajwala. who's willing to help answer the user's query with explanation.
    In your explanation, leverage your deep insurance expertise such as relevant insurance policies, complex coverage plans, or other pertinent insurance concepts.
    Use precise insurance terminology while still aiming to make the explanation clear and accessible to a general audience."""

    prompt = f"system\n{System}\nuser\Insurance Question: {Question}\nassistant\nInsurance Answer: "
    return prompt


def inference_loop(max_tokens=8025, top_k=15, n_gpu_layers=0, temperature=0.0, n_ctx=8192, n_threads=multiprocessing.cpu_count() - 1, n_batch=512):
    # Load the model
    print("Welcome to OpenInsuranceLLM Inference Loop:\n\n")

    llm_ctx = load_model(n_ctx, n_threads, n_batch, n_gpu_layers)
    print(f"OpenInsuranceLLM Q5_K_M model loaded successfully with n_batch={n_batch}!\n\nEnter your question (or type 'exit' to quit)\n")

    while True:
        Question = input("Raj: ").strip()
        if Question.lower() == "exit":
            print("Assistant: Good Bye!")
            break

        prompt = get_prompt(Question)

        start_time = time.time()
        response = llm_ctx(prompt, max_tokens=max_tokens, top_k=top_k, temperature=temperature)
        ntokens = response['usage']['completion_tokens']
        ntokens = 1 if ntokens == 0 else ntokens
        response = response['choices'][0]['text']
        response = dedent(response)
        execution_time = time.time() - start_time
        print(f"Assistant: {response}")
        print(f"tokens: {ntokens}")
        print(f"Time: {execution_time:.2f} s  Per Token: {(1.0 * execution_time / ntokens):.2f} s\n\n\n")

# default params set by me: inference_loop(max_tokens=8025, top_k=15, n_gpu_layers=0, temperature=0.0, n_ctx=8192, n_threads=multiprocessing.cpu_count() - 2, n_batch=512)
# multiprocessing.cpu_count() - 2 = 6
inference_loop(top_k=10, n_threads = 64, max_tokens=500)