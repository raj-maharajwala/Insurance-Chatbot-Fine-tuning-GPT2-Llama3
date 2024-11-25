# For Windows
python3 -m venv .venv_open_insurance_llm
.\venv\Scripts\activate

# For Mac/Linux
python3 -m venv .venv_open_insurance_llm
source .venv_open_insurance_llm/bin/activate

# Install with Metal support for Mac User
export FORCE_CMAKE=1
CMAKE_ARGS="-DGGML_METAL=on" pip install --upgrade --force-reinstall llama-cpp-python==0.3.2 --no-cache-dir

# Install Dependencies
pip install -r inference_requirements.txt

# Run Python Script
python3 model_inference_loop_quantized.py
