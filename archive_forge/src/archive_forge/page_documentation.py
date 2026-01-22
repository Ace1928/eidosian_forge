from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime

    **EXPERIMENTAL**

    Issued for every compilation cache generated. Is only available
    if Page.setGenerateCompilationCache is enabled.
    