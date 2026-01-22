from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig
def load_llama_from_config(config_path):
    model_config = LlamaConfig.from_pretrained(config_path)
    model = LlamaForCausalLM(config=model_config)
    return model