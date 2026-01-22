import json
from pathlib import Path
from typing import Any, Union
import yaml
from langchain_core.language_models.llms import BaseLLM
from langchain_community.llms import get_type_to_cls_dict
def load_llm_from_config(config: dict, **kwargs: Any) -> BaseLLM:
    """Load LLM from Config Dict."""
    if '_type' not in config:
        raise ValueError('Must specify an LLM Type in config')
    config_type = config.pop('_type')
    type_to_cls_dict = get_type_to_cls_dict()
    if config_type not in type_to_cls_dict:
        raise ValueError(f'Loading {config_type} LLM not supported')
    llm_cls = type_to_cls_dict[config_type]()
    load_kwargs = {}
    if _ALLOW_DANGEROUS_DESERIALIZATION_ARG in llm_cls.__fields__:
        load_kwargs[_ALLOW_DANGEROUS_DESERIALIZATION_ARG] = kwargs.get(_ALLOW_DANGEROUS_DESERIALIZATION_ARG, False)
    return llm_cls(**config, **load_kwargs)