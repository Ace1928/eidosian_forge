from __future__ import annotations
import logging
import os
from typing import Callable, Optional, Union
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from safetensors import SafetensorError, safe_open
from transformers.utils import cached_file
from transformers.utils.hub import get_checkpoint_shard_files
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
@torch.no_grad()
def replace_lora_weights_loftq(peft_model, model_path: Optional[str]=None, adapter_name: str='default', callback: Optional[Callable[[torch.nn.Module, str], bool]]=None):
    """
    Replace the LoRA weights of a model quantized with bitsandbytes, using the LoftQ technique.

    The replacement is done on the fly by loading in the non-quantized weights from a locally stored safetensors model
    file and initializing the LoRA weights such that the quantization error between the original and quantized weights
    is minimized.

    As lazy loading is not possible with pickle, normal PyTorch checkpoint files cannot be supported.

    Depending on the model size, calling this function may take some time to finish.

    Args:
        peft_model (`PeftModel`):
            The model to replace the weights of. Must be a quantized PEFT model with LoRA layers.
        model_path (`Optional[str]`):
            The path to the model safetensors file. If the model is a Hugging Face model, this will be inferred from
            the model's config. Otherwise, it must be provided.
        adapter_name (`str`):
            The name of the adapter to replace the weights of. The default adapter name is "default".
        callback (`Optional[Callable[[PeftModel, str], bool]]`):
            A callback function that will be called after each module is replaced. The callback function should take
            the model and the name of the current module as input and return a boolean indicating whether the
            replacement should be kept. If the callback returns False, the replacement will be rolled back. This can be
            very useful to confirm that the LoftQ initialization actually decreases the quantization error of the
            model. As an example, this callback could generate logits for given input and compare it with the logits
            from the original, non-quanitzed model with the same input, and only return `True` if there is an
            improvement. As this is a greedy optimization, it's possible that calling this function multiple times
            yields incremental improvements.
    """
    if not is_bnb_4bit_available():
        raise ValueError('bitsandbytes must be installed and the model must be quantized in 4bits.')
    from peft.tuners.lora import Linear4bit
    prefix = 'base_model.model.'
    any_match = False
    safetensor_loader = _SafetensorLoader(peft_model, model_path)
    for name, module in peft_model.named_modules():
        if not isinstance(module, Linear4bit):
            continue
        if not name.startswith(prefix):
            raise TypeError('The passed model does not appear to be a valid PeftModel')
        any_match = True
        name = name[len(prefix):]
        tensor = safetensor_loader.get_tensor(name + '.weight')
        reduced_rank = module.r[adapter_name]
        lora_A, lora_B = _loftq_init_new(module.weight, tensor, num_bits=4, reduced_rank=reduced_rank)
        if not callback:
            module.lora_A[adapter_name].weight.data = lora_A
            module.lora_B[adapter_name].weight.data = lora_B
            continue
        lora_A_before = module.lora_A[adapter_name].weight.data
        lora_B_before = module.lora_B[adapter_name].weight.data
        module.lora_A[adapter_name].weight.data = lora_A
        module.lora_B[adapter_name].weight.data = lora_B
        should_replace = callback(peft_model, name)
        if not should_replace:
            module.lora_A[adapter_name].weight.data = lora_A_before
            module.lora_B[adapter_name].weight.data = lora_B_before
        del lora_A_before, lora_B_before
    if not any_match:
        raise ValueError('No bnb LoRA module found on the model')