from __future__ import annotations
import logging # isort:skip
from abc import abstractmethod
from typing import Any
from ...core.has_props import abstract
from ...core.properties import (
from ...model import Model
def is_known(self, unit: str) -> bool:
    prefixes = ['Q', 'R', 'Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', '', 'd', 'c', 'm', 'µ', 'n', 'p', 'f', 'a', 'z', 'y', 'r', 'q']
    basis = {f'{prefix}{unit}' for prefix in prefixes}
    return unit in basis