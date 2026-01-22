import argparse
import transformers
import torch
def hf_add_tokens(model_path, output_dir, added_special_tokens):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    tokenizer.add_special_tokens({'additional_special_tokens': added_special_tokens})
    assert model.model.embed_tokens.weight.requires_grad
    assert model.lm_head.weight.requires_grad
    model.model.embed_tokens.weight = torch.nn.Parameter(add_tokens_to_embedding(added_special_tokens, model.model.embed_tokens.weight), requires_grad=True)
    model.lm_head.weight = torch.nn.Parameter(add_tokens_to_embedding(added_special_tokens, model.lm_head.weight), requires_grad=True)
    model.config.vocab_size += len(added_special_tokens)
    if 'mistral' in model_path.lower():
        assert model.config.max_position_embeddings == 32768
        model.config.max_position_embeddings = 8192
    print({k: v.shape for k, v in model.state_dict().items()})
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)