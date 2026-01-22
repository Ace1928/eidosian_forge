from __future__ import annotations
import asyncio
import base64
import copy
import json
import mimetypes
import os
import pkgutil
import secrets
import shutil
import tempfile
import warnings
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Literal, Optional, TypedDict
import fsspec.asyn
import httpx
import huggingface_hub
from huggingface_hub import SpaceStage
from websockets.legacy.protocol import WebSocketCommonProtocol
def encode_url_to_base64(url: str):
    resp = httpx.get(url)
    resp.raise_for_status()
    encoded_string = base64.b64encode(resp.content)
    base64_str = str(encoded_string, 'utf-8')
    mimetype = get_mimetype(url)
    return 'data:' + (mimetype if mimetype is not None else '') + ';base64,' + base64_str