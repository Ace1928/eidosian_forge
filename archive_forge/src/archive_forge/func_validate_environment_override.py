from __future__ import annotations
import logging
import sys
from typing import TYPE_CHECKING, Dict, Optional, Set
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.adapters.openai import convert_message_to_dict
from langchain_community.chat_models.openai import (
@root_validator(pre=True)
def validate_environment_override(cls, values: dict) -> dict:
    """Validate that api key and python package exists in environment."""
    values['openai_api_key'] = get_from_dict_or_env(values, 'everlyai_api_key', 'EVERLYAI_API_KEY')
    values['openai_api_base'] = DEFAULT_API_BASE
    try:
        import openai
    except ImportError as e:
        raise ValueError('Could not import openai python package. Please install it with `pip install openai`.') from e
    try:
        values['client'] = openai.ChatCompletion
    except AttributeError as exc:
        raise ValueError('`openai` has no `ChatCompletion` attribute, this is likely due to an old version of the openai package. Try upgrading it with `pip install --upgrade openai`.') from exc
    if 'model_name' not in values.keys():
        values['model_name'] = DEFAULT_MODEL
    model_name = values['model_name']
    available_models = cls.get_available_models()
    if model_name not in available_models:
        raise ValueError(f'Model name {model_name} not found in available models: {available_models}.')
    values['available_models'] = available_models
    return values