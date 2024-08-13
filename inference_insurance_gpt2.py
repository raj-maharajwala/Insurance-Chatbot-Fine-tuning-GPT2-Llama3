from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def generate_text(model_path, sequence, max_length):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)

ip = ''
while True:
  ip = input("User: ")
  if ip == "end":
    break
  ip = f"[Q] {ip}"
  ans = generate_text("model/Final_GPT2_QA_Finetuning", sequence=ip, max_length=100)
  start_index = ans.find('[A] ') + 4  # Finding the start index right after "[A]"
  extracted_text = ans[start_index:ans.find('[Q]', start_index) if ans.find('[Q]', start_index) != -1 else None].strip()
  print(f"Bot: {extracted_text}\n")