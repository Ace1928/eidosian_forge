import logging
import logging.handlers
from collections import defaultdict
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple, Type, Union
from docutils import nodes
from docutils.nodes import Node
from docutils.utils import get_source_line
from sphinx.errors import SphinxWarning
from sphinx.util.console import colorize
from sphinx.util.osutil import abspath
@contextmanager
def prefixed_warnings(prefix: str) -> Generator[None, None, None]:
    """Context manager to prepend prefix to all warning log records temporarily.

    For example::

        >>> with prefixed_warnings("prefix:"):
        >>>     logger.warning('Warning message!')  # => prefix: Warning message!

    .. versionadded:: 2.0
    """
    logger = logging.getLogger(NAMESPACE)
    warning_handler = None
    for handler in logger.handlers:
        if isinstance(handler, WarningStreamHandler):
            warning_handler = handler
            break
    else:
        yield
        return
    prefix_filter = None
    for _filter in warning_handler.filters:
        if isinstance(_filter, MessagePrefixFilter):
            prefix_filter = _filter
            break
    if prefix_filter:
        try:
            previous = prefix_filter.prefix
            prefix_filter.prefix = prefix
            yield
        finally:
            prefix_filter.prefix = previous
    else:
        prefix_filter = MessagePrefixFilter(prefix)
        try:
            warning_handler.addFilter(prefix_filter)
            yield
        finally:
            warning_handler.removeFilter(prefix_filter)