from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch

login("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")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it")

input_text = "hello."
input_ids = tokenizer(input_text, return_tensors="pt")

if torch.cuda.is_available():
    # 将模型移动到 GPU 上
    model.to('cuda')
    # 将输入数据移动到 GPU 上
    input_ids.to('cuda')

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
