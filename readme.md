# One-stop Insurance Chatbot

**Watch the Chatbot Demo:**   
![GIF](https://github.com/raj-maharajwala/Insurance-Chatbot-Fine-tuning-GPT2-Llama2/blob/main/video/InsuranceGPT_big.gif)

**1. Run the UI application and ask Queries:**<br>
Simply run the file `app.py` using below command 
```{python} 
python3 app.py 
```
<br>

**2. For inference purposes on the optimal model:**<br>
Simply run the `inference_gp2.py` using below command:
```{python} 
python3 inference_gp2.py 
```

Simply run the `inference_llama3.py` using below command:
```{python} 
python3 inference_llama3.py 
```
<br>

File `final_GPT2_finetuning.ipynb` contains data preparation, GPT-2 Model Training on most optimal parameters, Model evaluation, and Inference
n, Inference, Parameter Tuning Test, Testing Llama2 and Llama3 for future reference.

File `finetuning_Llama3_QLoRA_InsuranceQA.ipnb` contains data preparation, Llama-3 Model Training, Model evaluation, Inference, Parameter Tuning Test

File `GPT2_params_testing_Llama2.ipynb` contains data preparation, Model Training, Model evaluation
<br><br>

# Progress and More Information
<br>

1. Optimized and fine-tuned a decoder-only architecture Large Language Models GPT-2 and Llama-3, leveraging the InsuranceQA dataset to create a chatbot that provides clear information about insurance policies. Implemented data augmentation, 4-bit quantization using QLoRA configurations, and state-of-the-art PEFT methods to enhance model generalization and efficient resource utilization.

2. Achieved a Test set Perplexity score of 3.5 for GPT-2 Model and 1.6 for Llama-3 Model through hyperparameter tuning and model optimization, including use of AdaFactor optimizer, reduce lr on plateau scheduler, weight decay, and other training arguments during testing to ensure best possible model performance. Integrated fine-tuned GPT-2 model, and Llama-3 with a user-friendly web interface utilizing Flask.
<br><br>

The chatbot is trained on the InsuranceQA dataset to provide clear and accessible information to users, helping them navigate insurance policies and make informed decisions. The project explores various models such as DistilBERT, GPT-2, Llama-2, and Llama-3. Ultimately opting for decoder only Transformer based architecture GPT-2, Llama-3 due to its suitability for generating conversational responses. Significant efforts are made in data augmentation, model fine-tuning, and experimentation to optimize performance. The project's contributions include augmenting the InsuranceQA dataset, modifying and fine-tuning models, and thorough documentation of methodologies and results to facilitate reproducibility and further research in the field.

Finetuning played a crucial role in the project, enabling the customization of pre-trained language models to suit the intricacies of insurance-related queries.

In future, I am planning to improve my implementation and model performance as well as make the response multi-modal with Relevant reference media including relevant Yotube videos and relevant frame and timestamp of the video. Also, I will work with other models such as Llama 3 for finetuning on Insurance data. I have already tested reference video integration based on cosine similarity on audio transcript with timestamps.

