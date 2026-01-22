import re
import tokenize
from io import StringIO
from typing import Callable, List, Optional, Union, Generator, Tuple
import warnings
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyPressEvent
from prompt_toolkit.key_binding.bindings import named_commands as nc
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory, Suggestion
from prompt_toolkit.document import Document
from prompt_toolkit.history import History
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.layout.processors import (
from IPython.core.getipython import get_ipython
from IPython.utils.tokenutil import generate_tokens
from .filters import pass_through
def swap_autosuggestion_down(event: KeyPressEvent):
    """Get previous autosuggestion from history."""
    shell = get_ipython()
    provider = shell.auto_suggest
    if not isinstance(provider, NavigableAutoSuggestFromHistory):
        return
    return _swap_autosuggestion(buffer=event.current_buffer, provider=provider, direction_method=provider.down)