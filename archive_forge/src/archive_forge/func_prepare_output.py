import asyncio
import json
import warnings
from abc import ABC
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
from langchain_community.utilities.anthropic import (
@classmethod
def prepare_output(cls, provider: str, response: Any) -> dict:
    text = ''
    if provider == 'anthropic':
        response_body = json.loads(response.get('body').read().decode())
        if 'completion' in response_body:
            text = response_body.get('completion')
        elif 'content' in response_body:
            content = response_body.get('content')
            text = content[0].get('text')
    else:
        response_body = json.loads(response.get('body').read())
        if provider == 'ai21':
            text = response_body.get('completions')[0].get('data').get('text')
        elif provider == 'cohere':
            text = response_body.get('generations')[0].get('text')
        elif provider == 'meta':
            text = response_body.get('generation')
        elif provider == 'mistral':
            text = response_body.get('outputs')[0].get('text')
        else:
            text = response_body.get('results')[0].get('outputText')
    headers = response.get('ResponseMetadata', {}).get('HTTPHeaders', {})
    prompt_tokens = int(headers.get('x-amzn-bedrock-input-token-count', 0))
    completion_tokens = int(headers.get('x-amzn-bedrock-output-token-count', 0))
    return {'text': text, 'body': response_body, 'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': prompt_tokens + completion_tokens}}