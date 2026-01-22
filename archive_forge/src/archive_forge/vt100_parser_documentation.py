from __future__ import annotations
import re
from typing import Callable, Dict, Generator
from ..key_binding.key_processor import KeyPress
from ..keys import Keys
from .ansi_escape_sequences import ANSI_SEQUENCES

        Wrapper around ``feed`` and ``flush``.
        