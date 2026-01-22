from __future__ import annotations
import os
import pathlib
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from os import PathLike
from typing import (
from .. import to_thread
from ..abc import AsyncResource
def get_next_value() -> tuple[pathlib.Path, list[str], list[str]] | None:
    try:
        return next(gen)
    except StopIteration:
        return None