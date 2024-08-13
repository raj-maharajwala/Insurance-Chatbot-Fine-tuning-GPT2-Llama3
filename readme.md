# One-stop Insurance Chatbot

Inference Model: [🤗 OpenInsuranceLLM-Llama-3-8B-GGUF Model - hugging-face](https://huggingface.co/Raj-Maharajwala/OpenInsuranceLLM-Llama3-8B-GGUF)

**Watch the Chatbot Demo:**   
![GIF](Missing GIF Llama3 Stramlit Application)

![GIF](https://github.com/raj-maharajwala/Insurance-Chatbot-Fine-tuning-GPT2-Llama2/blob/main/video/InsuranceGPT_big.gif)


**1. Run the UI application and ask Queries:**<br>

Finetuned Llama3-8B: Run the file `app_llama3_streamlit.py` using below command 
```{bash}
streamlit run app_llama3_streamlit.py
```

Finetuned GPT2: Run the file `app_gpt2_flask.py` using below command
```{bash} 
python3 app_gpt2_flask.py 
```
<br>

**2. For inference purposes on the optimal model:**<br>
Simply run the `inference_insurance_gpt2.py` using below command:
```{bash} 
python3 inference_gp2.py 
```

Simply run the `inference_insurance_llama3_GGUF.py` using below command:
```{bash} 
python3 inference_insurance_llama3_GGUF.py 
```
<br>

File `data_processing.ipynb` contains initial data preprocessing steps from InsuranceQA to QA format for finetuning LLM models.

File `Finetuning_Llama3_LoRA_GGUF_latest.ipynb` contains Finetuning of Llama3 model using LoRA and also converting the Finetuned model to Q5_K_M, and Q4_K_M GGUF format for faster inference.

Final Model `Hugging Face link`: Raj-Maharajwala/OpenInsuranceLLM-Llama3-8B-GGUF

File `final_GPT2_finetuning.ipynb` contains data preparation, GPT-2 Model Training on most optimal parameters, Model evaluation, and Inference
n, Inference, Parameter Tuning Test, Testing Llama2 and Llama3 for future reference.

File `GPT2_params_testing_Llama2.ipynb` contains data preparation, Model Training, Model evaluation, and Testing Parameter
<br><br>

## 1. Finetuned on Base Model: Llama3-8B using LoRA (8-bit)

## 2. Dataset: InsuranceQA (Subset)
<br>

# Python Script for inference using llama_cpp_python
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

```python
import os
import psutil
import time
import glob
from pathlib import Path
from llama_cpp import Llama
from textwrap import dedent
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


def inference_loop(max_tokens=8025, top_k=15, n_gpu_layers=0, temperature=0.0, n_ctx=8192, n_threads=32, n_batch=512):
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
        response = dedent(response['choices'][0]['text'])
        execution_time = time.time() - start_time
        print(f"Assistant: {response}")
        print(f"tokens: {ntokens}")
        print(f"Time: {execution_time:.2f} s  Per Token: {(1.0*execution_time / ntokens):.2f} s  Token/sec: {round(1.0*ntokens/execution_time)} tokens/s\n\n\n")

#default params: inference_loop(max_tokens=8025, top_k=15, n_gpu_layers=0, temperature=0.0, n_ctx=8192, n_threads=32, n_batch=512):
inference_loop(top_k=10) 
```

# Progress and More Information
<br>

•	Optimized and Fine-tuned GPT-2 and Llama-3 models on InsuranceQA dataset, creating chatbot for clear insurance policy information. Performed data augmentation, 4-bit quantization using QLoRA configurations, and state-of-the-art PEFT methods to boost model generalization and efficient resource utilization. Integrated fine-tuned models with user-friendly web interface employing Flask.<br><br>
•	Achieved Test set Perplexity score of 3.5 for GPT-2 and 1.42 for Llama-3 model through hyperparameter tuning and model optimization, including use of PagedAdamW8bit optimizer, ReduceLROnPlateau scheduler, weight decay, and other training arguments during testing.<br>
•	Converted the Merged Finetuned Model to GGUF format (Q5_K_M, and Q4_K_M) for efficient inference for CPU devices using llama-cpp-python.

<br>

---
The chatbot is trained on the InsuranceQA dataset to provide clear and accessible information to users, helping them navigate insurance policies and make informed decisions. <br><br> Significant efforts are made in data augmentation, model fine-tuning, and experimentation to optimize performance. The project's contributions include augmenting the InsuranceQA dataset, modifying and fine-tuning models, and thorough documentation of methodologies and results to facilitate reproducibility and further research in the field.

Finetuning played a crucial role in the project, enabling the customization of pre-trained language models to suit the intricacies of insurance-related queries.


## Future Work:

1. I'm planning to enhance my search system by integrating Hybrid RAG, which combines Retrieval-Augmented Generation with Reciprocal Rank Fusion. This approach will allow me to leverage both keyword-based and semantic search capabilities, ensuring more accurate and contextually relevant results. By using Reciprocal Rank Fusion, I can merge results from different search methods, assigning scores based on rank positions to create a comprehensive and optimized ranked list.<br><br>
2. I'm looking to incorporate video resources by using audio transcript similarity matching. This will involve transcribing video content and aligning it with user queries to ensure the videos are contextually relevant. By extracting timestamps from these transcripts, I can direct users to specific segments of the videos that are most pertinent to their queries. This integration will enrich the search experience by providing not only text-based resources but also precise video content that addresses user needs effectively.

---
