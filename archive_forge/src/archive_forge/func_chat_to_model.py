import json
import time
from typing import AsyncIterable
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from mlflow.gateway.config import AnthropicConfig, RouteConfig
from mlflow.gateway.constants import (
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request, send_stream_request
from mlflow.gateway.schemas import chat, completions
@classmethod
def chat_to_model(cls, payload, config):
    key_mapping = {'stop': 'stop_sequences'}
    payload = rename_payload_keys(payload, key_mapping)
    if 'top_p' in payload and 'temperature' in payload:
        raise HTTPException(status_code=422, detail="Cannot set both 'temperature' and 'top_p' parameters.")
    max_tokens = payload.get('max_tokens', MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS)
    if max_tokens > MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS:
        raise HTTPException(status_code=422, detail=f'Invalid value for max_tokens: cannot exceed {MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS}.')
    payload['max_tokens'] = max_tokens
    if payload.pop('n', 1) != 1:
        raise HTTPException(status_code=422, detail="'n' must be '1' for the Anthropic provider. Received value: '{n}'.")
    system_messages = [m for m in payload['messages'] if m['role'] == 'system']
    if system_messages:
        payload['system'] = '\n'.join((m['content'] for m in system_messages))
    payload['messages'] = [m for m in payload['messages'] if m['role'] in ('user', 'assistant')]
    if 'temperature' in payload:
        payload['temperature'] = 0.5 * payload['temperature']
    return payload