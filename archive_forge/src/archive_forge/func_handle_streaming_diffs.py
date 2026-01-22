from __future__ import annotations
import copy
import hashlib
import inspect
import json
import os
import random
import secrets
import string
import sys
import threading
import time
import warnings
import webbrowser
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Literal, Sequence, cast
from urllib.parse import urlparse, urlunparse
import anyio
import fastapi
import httpx
from anyio import CapacityLimiter
from gradio_client import utils as client_utils
from gradio_client.documentation import document
from gradio import (
from gradio.blocks_events import BlocksEvents, BlocksMeta
from gradio.context import Context
from gradio.data_classes import FileData, GradioModel, GradioRootModel
from gradio.events import (
from gradio.exceptions import (
from gradio.helpers import create_tracker, skip, special_args
from gradio.state_holder import SessionState
from gradio.themes import Default as DefaultTheme
from gradio.themes import ThemeClass as Theme
from gradio.tunneling import (
from gradio.utils import (
def handle_streaming_diffs(self, fn_index: int, data: list, session_hash: str | None, run: int | None, final: bool, simple_format: bool=False) -> list:
    if session_hash is None or run is None:
        return data
    first_run = run not in self.pending_diff_streams[session_hash]
    if first_run:
        self.pending_diff_streams[session_hash][run] = [None] * len(data)
    last_diffs = self.pending_diff_streams[session_hash][run]
    for i in range(len(self.dependencies[fn_index]['outputs'])):
        if final:
            data[i] = last_diffs[i]
            continue
        if first_run:
            last_diffs[i] = data[i]
        else:
            prev_chunk = last_diffs[i]
            last_diffs[i] = data[i]
            if not simple_format:
                data[i] = utils.diff(prev_chunk, data[i])
    if final:
        del self.pending_diff_streams[session_hash][run]
    return data