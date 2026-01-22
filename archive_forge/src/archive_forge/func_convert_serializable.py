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
def convert_serializable(records: List[logging.LogRecord]) -> None:
    """Convert LogRecord serializable."""
    for r in records:
        r.msg = r.getMessage()
        r.args = ()
        location = getattr(r, 'location', None)
        if isinstance(location, nodes.Node):
            r.location = get_node_location(location)