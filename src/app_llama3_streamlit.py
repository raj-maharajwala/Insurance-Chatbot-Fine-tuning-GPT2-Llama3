import sys
import streamlit as st
from llama_cpp import Llama
import textwrap
import time
import os
import multiprocessing
from pathlib import Path
import base64

@st.cache_resource
def load_model(n_gpu_layers, n_ctx, n_threads, n_batch):
    parent_dir = os.path.dirname(os.getcwd())
    MODEL_DIR = os.path.join(parent_dir, 'gguf_dir') # GGUF files
    os.makedirs(MODEL_DIR, exist_ok=True)
    directory = Path(MODEL_DIR)
    model_1_path = str(list(directory.glob('openinsurancellm*Q5*.gguf'))[0])
    
    llm_ctx = Llama(model_path=model_1_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_batch=n_batch,
                    verbose=False)
    return llm_ctx

def get_prompt(Question, context=""):
    System = """You are an expert and experienced from the Insurance domain with extensive insurance knowledge and professional writter with all the insurance policies.
    Your name is OpenInsuranceLLM, and you were developed by Raj Maharajwala. who's willing to help answer the user's query with explanation.
    In your explanation, leverage your deep insurance expertise such as relevant insurance policies, complex coverage plans, or other pertinent insurance concepts.
    Use precise insurance terminology while still aiming to make the explanation clear and accessible to a general audience."""

    if context:
        prompt = f"system\n{System}\nuser\nPrevious context: {context}\nFollow-up Insurance question: {Question}\nassistant\nInsurance Answer: "
    else:
        prompt = f"system\n{System}\nuser\nInsurance Question: {Question}\nassistant\nInsurance Answer: "
    return prompt

st.title(":red[OpenInsuranceLLM Inference ]:heavy_dollar_sign::hospital:")

with open("../assets/logo.png", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

# Custom CSS for rotating logo and text styling
Sidebar_html = f"""
<style>
    @keyframes rotate-y {{
        0% {{ transform: rotateY(0deg); }}
        100% {{ transform: rotateY(360deg); }}
    }}
    .rotating-logo {{
        animation: rotate-y 8s linear infinite;
        transform-style: preserve-3d;
        width: 150px;
        height: 150px;
    }}
    .sidebar-text {{
        color: rgba(253,74,76,255);
        /*color: #e15642; --- orange*/
        /*color: #CC9933;  --- gold yellow*/
        font-size: 3em;
        margin-top: 5px;
        margin-bottom: -10px;
    }}
    .sidebar-subtext {{
        color: #D3D3D3;
        font-size: 1.0em;
        margin-bottom: -30px;
    }}
    .streamlit-expanderHeader {{
        font-size: 1.5em;
    }}
    .streamlit-expander {{
        width: 100% !important;
    }}
</style>
<img src="data:image/png;base64,{base64_image}" class="rotating-logo">
"""

st.sidebar.markdown(Sidebar_html, unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-text'>OpenInsuranceLLM</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-subtext'>Where Insurance Exploration Begins</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

with st.sidebar.expander("About"):
    st.markdown("""
    This is a chatbot interface for OpenInsuranceLLM, an AI assistant specialized in insurance domain. 
    Developed by Raj Maharajwala, it provides expert answers to insurance questions and other general questions suited for Llama-3 Model.

    ## How to use
    1. Adjust model parameters in the sidebar if needed.
    2. Type your insurance/general question in the chat input.
    3. The AI will generate a response based on its extensive insurance/general knowledge.

    Note: This is a demo version and should not replace professional insurance advice.
    """)

# Model parameters in a dropdown
with st.sidebar.expander("Model Parameters"):
    top_k = st.slider("Top K", 1, 100, 15)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
    max_tokens = st.slider("Max Tokens", 200, 10000, 800)
    n_gpu_layers = st.slider("GPU Layers", -1, 8, 0)
    n_ctx = st.slider("Context Size", 1000, 8192, 8192)
    n_threads = st.slider("Threads", 1, 64, multiprocessing.cpu_count() - 1)
    n_batch = st.slider("Batch Size", 1, 1024, 512)

# Load the model
model = load_model(n_gpu_layers, n_ctx, n_threads, n_batch)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.context = ""

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if question := st.chat_input("Ask anything in Insurance domain (or general questions) ..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        start_time = time.time()
        llm_prompt = get_prompt(question, st.session_state.context)
        response = model(llm_prompt, max_tokens=max_tokens, top_k=top_k, temperature=temperature)
        ntokens = response['usage']['completion_tokens']
        ntokens = 1 if ntokens == 0 else ntokens
        response_text = response['choices'][0]['text'] 
        execution_time = time.time() - start_time
        response_text += f"<br>Time: {execution_time:.2f} s Per Token: {(1.0*execution_time/ntokens):.2f} s  Token/sec: {(1.0*ntokens/execution_time):.2f} s"

        full_response = ""
        for chunk in textwrap.wrap(response_text, width=60):
            full_response += chunk + "<br>"  
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
        message_placeholder.markdown(full_response, unsafe_allow_html=True)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Update context for follow-up questions
    st.session_state.context += f"{question}\n" # Only Added Question not Answer or System message again for follow up questions