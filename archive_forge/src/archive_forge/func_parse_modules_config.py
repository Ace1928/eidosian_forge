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
def parse_modules_config(data: t.Any) -> ModulesConfig:
    """Parse the given dictionary as module config and return it."""
    if not isinstance(data, dict):
        raise Exception('config must be type `dict` not `%s`' % type(data))
    python_requires = data.get('python_requires', MISSING)
    if python_requires == MISSING:
        raise KeyError('python_requires is required')
    return ModulesConfig(python_requires=python_requires, python_versions=parse_python_requires(python_requires), controller_only=python_requires == 'controller')