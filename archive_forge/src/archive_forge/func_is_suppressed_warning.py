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
def is_suppressed_warning(type: str, subtype: str, suppress_warnings: List[str]) -> bool:
    """Check whether the warning is suppressed or not."""
    if type is None:
        return False
    subtarget: Optional[str]
    for warning_type in suppress_warnings:
        if '.' in warning_type:
            target, subtarget = warning_type.split('.', 1)
        else:
            target, subtarget = (warning_type, None)
        if target == type and subtarget in (None, subtype, '*'):
            return True
    return False