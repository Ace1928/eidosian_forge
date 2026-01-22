from __future__ import annotations
import logging # isort:skip
import weakref
from collections import defaultdict
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable
from ..core.enums import HoldPolicy, HoldPolicyType
from ..events import (
from ..model import Model
from ..models.callbacks import Callback as JSEventCallback
from ..util.callback_manager import _check_callback
from .events import (
from .locking import UnlockedDocumentProxy
def invoke_with_curdoc(doc: Document, f: Callable[[], None]) -> None:
    from ..io.doc import patch_curdoc
    curdoc: Document | UnlockedDocumentProxy = UnlockedDocumentProxy(doc) if getattr(f, 'nolock', False) else doc
    with patch_curdoc(curdoc):
        return f()