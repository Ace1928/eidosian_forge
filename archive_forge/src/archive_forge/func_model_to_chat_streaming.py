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
def model_to_chat_streaming(cls, resp, config):
    content = resp.get('delta') or resp.get('content_block') or {}
    if (stop_reason := content.get('stop_reason')) is not None:
        stop_reason = 'length' if stop_reason == 'max_tokens' else 'stop'
    return chat.StreamResponsePayload(id=resp['id'], created=int(time.time()), model=resp['model'], choices=[chat.StreamChoice(index=resp['index'], finish_reason=stop_reason, delta=chat.StreamDelta(role=None, content=content.get('text')))])