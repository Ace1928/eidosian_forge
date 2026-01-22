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
@validator('http_client', always=True)
def validate_client(cls, field_value: Any, values: Dict) -> AzureMLEndpointClient:
    """Validate that api key and python package exists in environment."""
    endpoint_url = values.get('endpoint_url')
    endpoint_key = values.get('endpoint_api_key')
    deployment_name = values.get('deployment_name')
    timeout = values.get('timeout', DEFAULT_TIMEOUT)
    http_client = AzureMLEndpointClient(endpoint_url, endpoint_key.get_secret_value(), deployment_name, timeout)
    return http_client