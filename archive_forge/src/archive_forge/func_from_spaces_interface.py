from __future__ import annotations
import json
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable
import httpx
import huggingface_hub
from gradio_client import Client
from gradio_client.client import Endpoint
from gradio_client.documentation import document
from packaging import version
import gradio
from gradio import components, external_utils, utils
from gradio.context import Context
from gradio.exceptions import (
from gradio.processing_utils import save_base64_to_cache, to_binary
def from_spaces_interface(model_name: str, config: dict, alias: str | None, hf_token: str | None, iframe_url: str, **kwargs) -> Interface:
    config = external_utils.streamline_spaces_interface(config)
    api_url = f'{iframe_url}/api/predict/'
    headers = {'Content-Type': 'application/json'}
    if hf_token is not None:
        headers['Authorization'] = f'Bearer {hf_token}'

    def fn(*data):
        data = json.dumps({'data': data})
        response = httpx.post(api_url, headers=headers, data=data)
        result = json.loads(response.content.decode('utf-8'))
        if 'error' in result and '429' in result['error']:
            raise TooManyRequestsError('Too many requests to the Hugging Face API')
        try:
            output = result['data']
        except KeyError as ke:
            raise KeyError(f"Could not find 'data' key in response from external Space. Response received: {result}") from ke
        if len(config['outputs']) == 1:
            output = output[0]
        if len(config['outputs']) == 1 and isinstance(output, list):
            output = output[0]
        return output
    fn.__name__ = alias if alias is not None else model_name
    config['fn'] = fn
    kwargs = dict(config, **kwargs)
    kwargs['_api_mode'] = True
    interface = gradio.Interface(**kwargs)
    return interface