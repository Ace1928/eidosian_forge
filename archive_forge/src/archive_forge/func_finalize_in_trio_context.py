from __future__ import annotations
import logging
import sys
import warnings
import weakref
from typing import TYPE_CHECKING, NoReturn
import attrs
from .. import _core
from .._util import name_asyncgen
from . import _run
def finalize_in_trio_context(agen: AsyncGeneratorType[object, NoReturn], agen_name: str) -> None:
    try:
        runner.spawn_system_task(self._finalize_one, agen, agen_name, name=f'close asyncgen {agen_name} (abandoned)')
    except RuntimeError:
        self.trailing_needs_finalize.add(agen)