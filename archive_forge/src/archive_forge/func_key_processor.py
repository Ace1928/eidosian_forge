from __future__ import annotations
import weakref
from asyncio import Task, sleep
from collections import deque
from typing import TYPE_CHECKING, Any, Generator
from prompt_toolkit.application.current import get_app
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters.app import vi_navigation_mode
from prompt_toolkit.keys import Keys
from prompt_toolkit.utils import Event
from .key_bindings import Binding, KeyBindingsBase
@property
def key_processor(self) -> KeyProcessor:
    processor = self._key_processor_ref()
    if processor is None:
        raise Exception('KeyProcessor was lost. This should not happen.')
    return processor