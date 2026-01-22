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
@validator('endpoint_api_type')
def validate_endpoint_api_type(cls, field_value: Any, values: Dict) -> AzureMLEndpointApiType:
    """Validate that endpoint api type is compatible with the URL format."""
    endpoint_url = values.get('endpoint_url')
    if (field_value == AzureMLEndpointApiType.dedicated or field_value == AzureMLEndpointApiType.realtime) and (not endpoint_url.endswith('/score')):
        raise ValueError("Endpoints of type `dedicated` should follow the format `https://<your-endpoint>.<your_region>.inference.ml.azure.com/score`. If your endpoint URL ends with `/v1/completions` or`/v1/chat/completions`, use `endpoint_api_type='serverless'` instead.")
    if field_value == AzureMLEndpointApiType.serverless and (not (endpoint_url.endswith('/v1/completions') or endpoint_url.endswith('/v1/chat/completions'))):
        raise ValueError('Endpoints of type `serverless` should follow the format `https://<your-endpoint>.<your_region>.inference.ml.azure.com/v1/chat/completions` or `https://<your-endpoint>.<your_region>.inference.ml.azure.com/v1/chat/completions`')
    return field_value