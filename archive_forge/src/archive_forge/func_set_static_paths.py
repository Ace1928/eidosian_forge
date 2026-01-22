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
@document()
def set_static_paths(paths: list[str | Path]) -> None:
    """
    Set the static paths to be served by the gradio app.

    Static files are not moved to the gradio cache and are served directly from the file system.
    This function is useful when you want to serve files that you know will not be modified during the lifetime of the gradio app (like files used in gr.Examples).
    By setting static paths, your app will launch faster and it will consume less disk space.
    Calling this function will set the static paths for all gradio applications defined in the same interpreter session until it is called again or the session ends.
    To clear out the static paths, call this function with an empty list.

    Parameters:
        paths: List of filepaths or directory names to be served by the gradio app. If it is a directory name, ALL files located within that directory will be considered static and not moved to the gradio cache. This also means that ALL files in that directory will be accessible over the network.
    Example:
        import gradio as gr

        # Paths can be a list of strings or pathlib.Path objects
        # corresponding to filenames or directories.
        gr.set_static_paths(paths=["test/test_files/"])

        # The example files and the default value of the input
        # will not be copied to the gradio cache and will be served directly.
        demo = gr.Interface(
            lambda s: s.rotate(45),
            gr.Image(value="test/test_files/cheetah1.jpg", type="pil"),
            gr.Image(),
            examples=["test/test_files/bus.png"],
        )

        demo.launch()
    """
    from gradio.data_classes import _StaticFiles
    _StaticFiles.all_paths.extend([Path(p).resolve() for p in paths])