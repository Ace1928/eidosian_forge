from __future__ import annotations
import ast
import base64
import builtins  # Explicitly use builtins.set as 'set' will be shadowed by a function
import json
import os
import pathlib
import site
import sys
import threading
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal, overload
import yaml
from dask.typing import no_default
def paths_containing_key(key: str, paths: Sequence[str]=paths) -> Iterator[pathlib.Path]:
    """
    Generator yielding paths which contain the given key.
    """
    for path_ in paths:
        for path, config in collect_yaml([path_], return_paths=True):
            try:
                get(key, config=config)
            except KeyError:
                continue
            else:
                yield pathlib.Path(path)