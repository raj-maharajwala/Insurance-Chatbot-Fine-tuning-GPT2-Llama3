from flask import Flask, render_template, request, redirect, url_for, session
from transformers import GPT2LMHeadModel, GPT2Tokenizer
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

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

@app.route('/', methods=['GET', 'POST'])
def index():
    response = None
    if request.method == 'POST':
        query = request.form.get('query')
        print("-------",query)
        if 'history' not in session:
            session['history'] = []
        session['history'].append(query)
        session.modified = True
        ip = f"[Q] {query}"
        ans = generate_text("model/Final_GPT2_QA_Finetuning", sequence=ip, max_length=200)
        start_index = ans.find('[A] ') + 4  # Finding the start index right after "[A]"
        response = ans[start_index:ans.find('[Q]', start_index) if ans.find('[Q]', start_index) != -1 else None].strip()
        return render_template('index.html', history=session.get('history', []), query = query, response=response)
    return render_template('index.html', history=session.get('history', []), query = "Eager to explore the world of insurance? Ask me anything!", response="Welcome to InsuranceGPT, your trusted companion for all insurance-related inquiries. Ask away, and let's navigate the world of insurance together!")

if __name__ == '__main__':
    app.run(debug=True)
