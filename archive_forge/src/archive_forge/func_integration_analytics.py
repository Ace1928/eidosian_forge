from __future__ import annotations
import asyncio
import json
import os
import threading
import urllib.parse
import warnings
from typing import Any
import httpx
from packaging.version import Version
import gradio
from gradio import wasm_utils
from gradio.context import Context
from gradio.utils import get_package_version
def integration_analytics(data: dict[str, Any]) -> None:
    if not analytics_enabled():
        return
    _do_analytics_request(url=f'{ANALYTICS_URL}gradio-integration-analytics/', data=data)