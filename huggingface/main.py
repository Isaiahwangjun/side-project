# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("taide/TAIDE-LX-7B-Chat")
model = AutoModelForCausalLM.from_pretrained("taide/TAIDE-LX-7B-Chat")

chat = [
    {
        "role": "user",
        "content": "自學python的書籍，條列式列出"
    },
]
prompt = tokenizer.apply_chat_template(chat)
print(prompt)
