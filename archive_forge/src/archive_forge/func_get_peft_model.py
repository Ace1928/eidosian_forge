from __future__ import annotations
from typing import TYPE_CHECKING, Any
import torch
from .config import PeftConfig
from .mixed_model import PeftMixedModel
from .peft_model import (
from .tuners import (
from .utils import _prepare_prompt_learning_config
def get_peft_model(model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str='default', mixed: bool=False) -> PeftModel | PeftMixedModel:
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]):
            Model to be wrapped.
        peft_config ([`PeftConfig`]):
            Configuration object containing the parameters of the Peft model.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
        mixed (`bool`, `optional`, defaults to `False`):
            Whether to allow mixing different (compatible) adapter types.
    """
    model_config = getattr(model, 'config', {'model_type': 'custom'})
    if hasattr(model_config, 'to_dict'):
        model_config = model_config.to_dict()
    peft_config.base_model_name_or_path = model.__dict__.get('name_or_path', None)
    if mixed:
        return PeftMixedModel(model, peft_config, adapter_name=adapter_name)
    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and (not peft_config.is_prompt_learning):
        return PeftModel(model, peft_config, adapter_name=adapter_name)
    if peft_config.is_prompt_learning:
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model, peft_config, adapter_name=adapter_name)