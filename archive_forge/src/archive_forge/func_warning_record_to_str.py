from contextlib import contextmanager
import sys
from typing import Generator
from typing import Literal
from typing import Optional
import warnings
from _pytest.config import apply_warning_filters
from _pytest.config import Config
from _pytest.config import parse_warning_filter
from _pytest.main import Session
from _pytest.nodes import Item
from _pytest.terminal import TerminalReporter
import pytest
def warning_record_to_str(warning_message: warnings.WarningMessage) -> str:
    """Convert a warnings.WarningMessage to a string."""
    warn_msg = warning_message.message
    msg = warnings.formatwarning(str(warn_msg), warning_message.category, warning_message.filename, warning_message.lineno, warning_message.line)
    if warning_message.source is not None:
        try:
            import tracemalloc
        except ImportError:
            pass
        else:
            tb = tracemalloc.get_object_traceback(warning_message.source)
            if tb is not None:
                formatted_tb = '\n'.join(tb.format())
                msg += f'\nObject allocated at:\n{formatted_tb}'
            else:
                url = 'https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings'
                msg += 'Enable tracemalloc to get traceback where the object was allocated.\n'
                msg += f'See {url} for more info.'
    return msg