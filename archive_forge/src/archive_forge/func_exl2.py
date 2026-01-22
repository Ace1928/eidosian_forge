from typing import TYPE_CHECKING, Optional
import torch
from .transformers import TransformerTokenizer
def exl2(model_path: str, device: Optional[str]=None, model_kwargs: dict={}, tokenizer_kwargs: dict={}):
    try:
        from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError('The `exllamav2` library needs to be installed in order to use `exllamav2` models.')
    config = ExLlamaV2Config()
    config.model_dir = model_path
    config.prepare()
    config.max_seq_len = model_kwargs.pop('max_seq_len', config.max_seq_len)
    config.scale_pos_emb = model_kwargs.pop('scale_pos_emb', config.scale_pos_emb)
    config.scale_alpha_value = model_kwargs.pop('scale_alpha_value', config.scale_alpha_value)
    config.no_flash_attn = model_kwargs.pop('no_flash_attn', config.no_flash_attn)
    config.num_experts_per_token = int(model_kwargs.pop('num_experts_per_token', config.num_experts_per_token))
    model = ExLlamaV2(config)
    split = None
    if 'gpu_split' in model_kwargs.keys():
        split = [float(alloc) for alloc in model_kwargs['gpu_split'].split(',')]
    model.load(split)
    tokenizer_kwargs.setdefault('padding_side', 'left')
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    cache = ExLlamaV2Cache(model)
    return ExLlamaV2Model(model, tokenizer, device, cache)