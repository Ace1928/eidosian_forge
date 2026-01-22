from __future__ import annotations as _annotations
import json
import os
import sys
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Sequence, Tuple, Union, cast
from dotenv import dotenv_values
from pydantic import AliasChoices, AliasPath, BaseModel, Json
from pydantic._internal._typing_extra import WithArgsTypes, origin_is_union
from pydantic._internal._utils import deep_update, is_model_class, lenient_issubclass
from pydantic.fields import FieldInfo
from typing_extensions import get_args, get_origin
from pydantic_settings.utils import path_type_label
def import_toml() -> None:
    global tomli
    global tomllib
    if sys.version_info < (3, 11):
        if tomli is not None:
            return
        try:
            import tomli
        except ImportError as e:
            raise ImportError('tomli is not installed, run `pip install pydantic-settings[toml]`') from e
    else:
        if tomllib is not None:
            return
        import tomllib