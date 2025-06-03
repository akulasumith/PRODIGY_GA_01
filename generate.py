from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token to eos token to avoid padding errors
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Your topic prompt
prompt = "Imagine a future where humans and robots live in harmony"

# Tokenize input with attention mask
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

# Generate text
outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=150,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🌱 Generated Text\n")
print(generated_text)
