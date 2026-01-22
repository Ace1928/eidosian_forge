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
def custom_post_binary(data):
    data = to_binary({'path': data})
    response = httpx.request('POST', api_url, headers=headers, content=data)
    return save_base64_to_cache(external_utils.encode_to_base64(response), cache_dir=GRADIO_CACHE)