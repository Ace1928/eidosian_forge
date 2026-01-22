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
@contextmanager
def catch_warnings_for_item(config: Config, ihook, when: Literal['config', 'collect', 'runtest'], item: Optional[Item]) -> Generator[None, None, None]:
    """Context manager that catches warnings generated in the contained execution block.

    ``item`` can be None if we are not in the context of an item execution.

    Each warning captured triggers the ``pytest_warning_recorded`` hook.
    """
    config_filters = config.getini('filterwarnings')
    cmdline_filters = config.known_args_namespace.pythonwarnings or []
    with warnings.catch_warnings(record=True) as log:
        assert log is not None
        if not sys.warnoptions:
            warnings.filterwarnings('always', category=DeprecationWarning)
            warnings.filterwarnings('always', category=PendingDeprecationWarning)
        apply_warning_filters(config_filters, cmdline_filters)
        nodeid = '' if item is None else item.nodeid
        if item is not None:
            for mark in item.iter_markers(name='filterwarnings'):
                for arg in mark.args:
                    warnings.filterwarnings(*parse_warning_filter(arg, escape=False))
        try:
            yield
        finally:
            for warning_message in log:
                ihook.pytest_warning_recorded.call_historic(kwargs=dict(warning_message=warning_message, nodeid=nodeid, when=when, location=None))