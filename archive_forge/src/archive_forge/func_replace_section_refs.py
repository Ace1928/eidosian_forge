import copy
import inspect
import io
import re
import warnings
from configparser import (
from dataclasses import dataclass
from pathlib import Path
from types import GeneratorType
from typing import (
import srsly
from .util import SimpleFrozenDict, SimpleFrozenList  # noqa: F401
def replace_section_refs(self, config: Union[Dict[str, Any], 'Config'], parent: str='') -> None:
    """Replace references to section blocks in the final config."""
    for key, value in config.items():
        key_parent = f'{parent}.{key}'.strip('.')
        if isinstance(value, dict):
            self.replace_section_refs(value, parent=key_parent)
        elif isinstance(value, list):
            config[key] = [self._get_section_ref(v, parent=[parent, key]) for v in value]
        else:
            config[key] = self._get_section_ref(value, parent=[parent, key])