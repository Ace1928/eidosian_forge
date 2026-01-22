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
def delete_files_created_by_app(blocks: Blocks, age: int | None) -> None:
    """Delete files that are older than age. If age is None, delete all files."""
    dont_delete = set()
    for component in blocks.blocks.values():
        dont_delete.update(getattr(component, 'keep_in_cache', set()))
    for temp_set in blocks.temp_file_sets:
        to_remove = set()
        for file in temp_set:
            if file in dont_delete:
                continue
            try:
                file_path = Path(file)
                modified_time = datetime.fromtimestamp(file_path.lstat().st_ctime)
                if age is None or (datetime.now() - modified_time).seconds > age:
                    os.remove(file)
                    to_remove.add(file)
            except FileNotFoundError:
                continue
        temp_set -= to_remove