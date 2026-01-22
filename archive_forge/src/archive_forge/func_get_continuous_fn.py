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
def get_continuous_fn(fn: Callable, every: float) -> Callable:

    async def continuous_coro(*args):
        while True:
            output = fn(*args)
            if isinstance(output, GeneratorType):
                for item in output:
                    yield item
            elif isinstance(output, AsyncGeneratorType):
                async for item in output:
                    yield item
            elif inspect.isawaitable(output):
                yield (await output)
            else:
                yield output
            await asyncio.sleep(every)
    return continuous_coro