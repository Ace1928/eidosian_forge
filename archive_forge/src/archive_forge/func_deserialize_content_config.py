from __future__ import annotations
import os
import pickle
import typing as t
from .constants import (
from .compat.packaging import (
from .compat.yaml import (
from .io import (
from .util import (
from .data import (
from .config import (
def deserialize_content_config(path: str) -> ContentConfig:
    """Deserialize content config from the path."""
    with open_binary_file(path) as config_file:
        return pickle.load(config_file)