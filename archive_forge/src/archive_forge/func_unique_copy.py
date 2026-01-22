from __future__ import annotations
import pathlib
import secrets
import shutil
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
from fastapi import Request
from gradio_client.utils import traverse
from . import wasm_utils
def unique_copy(obj: dict):
    data = FileData(**obj)
    return data._copy_to_dir(str(pathlib.Path(dir / secrets.token_hex(10)))).model_dump()