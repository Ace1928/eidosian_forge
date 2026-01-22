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
def check_function_inputs_match(fn: Callable, inputs: list, inputs_as_dict: bool):
    """
    Checks if the input component set matches the function
    Returns: None if valid or if the function does not have a signature (e.g. is a built in),
    or a string error message if mismatch
    """
    try:
        signature = inspect.signature(fn)
    except ValueError:
        return None
    parameter_types = get_type_hints(fn)
    min_args = 0
    max_args = 0
    infinity = -1
    for name, param in signature.parameters.items():
        has_default = param.default != param.empty
        if param.kind in [param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD]:
            if not is_special_typed_parameter(name, parameter_types):
                if not has_default:
                    min_args += 1
                max_args += 1
        elif param.kind == param.VAR_POSITIONAL:
            max_args = infinity
        elif param.kind == param.KEYWORD_ONLY and (not has_default):
            return f'Keyword-only args must have default values for function {fn}'
    arg_count = 1 if inputs_as_dict else len(inputs)
    if min_args == max_args and max_args != arg_count:
        warnings.warn(f'Expected {max_args} arguments for function {fn}, received {arg_count}.')
    if arg_count < min_args:
        warnings.warn(f'Expected at least {min_args} arguments for function {fn}, received {arg_count}.')
    if max_args != infinity and arg_count > max_args:
        warnings.warn(f'Expected maximum {max_args} arguments for function {fn}, received {arg_count}.')