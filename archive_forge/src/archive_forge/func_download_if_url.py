from __future__ import annotations
import ast
import asyncio
import copy
import dataclasses
import functools
import importlib
import importlib.util
import inspect
import json
import json.decoder
import os
import pkgutil
import re
import sys
import tempfile
import threading
import time
import traceback
import typing
import urllib.parse
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from io import BytesIO
from numbers import Number
from pathlib import Path
from types import AsyncGeneratorType, GeneratorType, ModuleType
from typing import (
import anyio
import gradio_client.utils as client_utils
import httpx
from gradio_client.documentation import document
from typing_extensions import ParamSpec
import gradio
from gradio.context import Context
from gradio.data_classes import FileData
from gradio.strings import en
def download_if_url(article: str) -> str:
    try:
        result = urllib.parse.urlparse(article)
        is_url = all([result.scheme, result.netloc, result.path])
        is_url = is_url and result.scheme in ['http', 'https']
    except ValueError:
        is_url = False
    if not is_url:
        return article
    try:
        response = httpx.get(article, timeout=3)
        if response.status_code == httpx.codes.OK:
            article = response.text
    except (httpx.InvalidURL, httpx.RequestError, httpx.TimeoutException):
        pass
    return article