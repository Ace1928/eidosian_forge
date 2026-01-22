import json
import urllib.request
import warnings
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator, validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
def format_request_payload(self, prompt: str, model_kwargs: Dict, api_type: AzureMLEndpointApiType) -> bytes:
    """Formats the request according to the chosen api"""
    prompt = ContentFormatterBase.escape_special_characters(prompt)
    if api_type in [AzureMLEndpointApiType.dedicated, AzureMLEndpointApiType.realtime]:
        request_payload = json.dumps({'input_data': {'input_string': [f'"{prompt}"'], 'parameters': model_kwargs}})
    elif api_type == AzureMLEndpointApiType.serverless:
        request_payload = json.dumps({'prompt': prompt, **model_kwargs})
    else:
        raise ValueError(f'`api_type` {api_type} is not supported by this formatter')
    return str.encode(request_payload)