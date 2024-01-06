import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def get_completion(query: str, model, tokenizer) -> str:
  device = "cuda:0"
  prompt_template = """
  Below is an instruction that describes a task. Write a response in english that appropriately completes the request.
  ### Question:
  {query}

  ### Answer in english:
  """
  prompt = prompt_template.format(query=query)
  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
  model_inputs = encodeds.to(device)
  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids)
  return (decoded[0])

# Define custom quantization configuration for BitsAndBytes (BNB) quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Load the model with 4-bit quantization
    bnb_4bit_use_double_quant=True,       # Use double quantization for 4-bit weights
    bnb_4bit_quant_type="nf4",           # Use nf4 quantization method
    bnb_4bit_compute_dtype=torch.bfloat16 # Compute with 4-bit quantized weights in bfloat16 data type
)

# Specify the pre-trained model identifier
model_id = "/home/niwang/Mistral-7B-v0.1"

# Load the pre-trained model with the specified quantization configuration
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

# Load the tokenizer for the same pre-trained model and add an end-of-sequence token
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

result = get_completion(query="Will capital gains affect my tax bracket?", model=model, tokenizer=tokenizer)
print(result)
