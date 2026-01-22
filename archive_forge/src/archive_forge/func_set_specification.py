from __future__ import annotations
import inspect
import sys
import warnings
from types import ModuleType
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import Final
from typing import final
from typing import Generator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypedDict
from typing import TypeVar
from typing import Union
from ._result import Result
def set_specification(self, specmodule_or_class: _Namespace, spec_opts: HookspecOpts) -> None:
    if self.spec is not None:
        raise ValueError(f'Hook {self.spec.name!r} is already registered within namespace {self.spec.namespace}')
    self.spec = HookSpec(specmodule_or_class, self.name, spec_opts)
    if spec_opts.get('historic'):
        self._call_history = []