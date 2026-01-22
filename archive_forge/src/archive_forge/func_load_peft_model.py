from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model