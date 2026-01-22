import fnmatch
import importlib.machinery
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Generator, Sequence, Iterable, Union
from .line import (
def module_matches(self, cw: str, prefix: str='') -> Set[str]:
    """Modules names to replace cw with"""
    full = f'{prefix}.{cw}' if prefix else cw
    matches = (name for name in self.modules if name.startswith(full) and name.find('.', len(full)) == -1)
    if prefix:
        return {match[len(prefix) + 1:] for match in matches}
    else:
        return set(matches)