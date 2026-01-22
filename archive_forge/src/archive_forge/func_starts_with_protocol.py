from __future__ import annotations
import asyncio
import functools
import hashlib
import hmac
import json
import os
import re
import shutil
import sys
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass as python_dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, _TemporaryFileWrapper
from typing import (
from urllib.parse import urlparse
import anyio
import fastapi
import gradio_client.utils as client_utils
import httpx
import multipart
from gradio_client.documentation import document
from multipart.multipart import parse_options_header
from starlette.datastructures import FormData, Headers, MutableHeaders, UploadFile
from starlette.formparsers import MultiPartException, MultipartPart
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from gradio import processing_utils, utils
from gradio.data_classes import PredictBody
from gradio.exceptions import Error
from gradio.helpers import EventData
from gradio.state_holder import SessionState
def starts_with_protocol(string: str) -> bool:
    """This regex matches strings that start with a scheme (one or more characters not including colon, slash, or space)
    followed by ://, or start with just //, \\/, /\\, or \\ as they are interpreted as SMB paths on Windows.
    """
    pattern = '^(?:[a-zA-Z][a-zA-Z0-9+\\-.]*://|//|\\\\\\\\|\\\\/|/\\\\)'
    return re.match(pattern, string) is not None