# One-stop Insurance Chatbot

**Watch the Chatbot Demo:**  
<img src="[https://github.com/raj-maharajwala/Insurance-Chatbot-Fine-tuning-GPT2-Llama2/assets/95955903/7f0f0b78-dd78-4928-91f7-ef46905d9c27](https://drive.google.com/drive/u/0/folders/1iTVCHQADieQNnJlWENPMtAuy_acRBs-t)" alt="InsuranceGPT Demo" width="1000">

**1. Run the UI application and ask Queries:**<br>
Simply run the file `app.py` using below command 
```{python} 
python3 app.py 
```
<br>

**2. For inference purposes on the optimal model:**<br>
Simply run the `inference.py` using below command:
```{python} 
python3 inference.py 
```
<br>

File `final_GPT2_finetuning.ipynb` contains data preparation, Model Training on most optimal parameters, Model evaluation, and Inference

File `GPT2_params_testing_Llama2_Llama3.ipynb` contains data preparation, Model Training, Model evaluation, Inference, Parameter Tuning Test, Testing Llama2 and Llama3 for future reference.
<br><br>

# Progress and More Information

The chatbot is trained on the InsuranceQA dataset to provide clear and accessible information to users, helping them navigate insurance policies and make informed decisions. The project explores various models such as BERT, DistilBERT, and GPT-2, ultimately opting for GPT-2 due to its suitability for generating conversational responses. Significant efforts are made in data augmentation, model fine-tuning, and experimentation to optimize performance. The project's contributions include augmenting the InsuranceQA dataset, modifying and fine-tuning models, and thorough documentation of methodologies and results to facilitate reproducibility and further research in the field.

Finetuning played a crucial role in the project, enabling the customization of pre-trained language models to suit the intricacies of insurance-related queries. Initially exploring models like BERT and DistilBERT, the project ultimately opted for GPT-2, a decoder-only model, for its ability to generate conversational responses. The GPT-2 model undergoes extensive fine-tuning on the InsuranceQA dataset, augmented through paraphrasing techniques, to improve its performance in understanding and responding to user queries about insurance policies. Through meticulous experimentation and optimization of hyperparameters, the fine-tuned GPT-2 model achieves promising results, demonstrating its effectiveness in providing accessible and informative responses to users navigating the complexities of insurance products.

In future, I am planning to improve my implementation and model performance as well as make the response multi-modal with Relevant reference media including relevant Yotube videos and relevant frame and timestamp of the video. Also, I will work with other models such as Llama 3 for finetuning on Insurance data. I have already tested reference video integration based on cosine similarity on audio transcript with timestamps.

