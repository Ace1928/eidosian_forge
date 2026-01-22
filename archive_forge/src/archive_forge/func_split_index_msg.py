import functools
import hashlib
import os
import posixpath
import re
import sys
import tempfile
import traceback
import warnings
from datetime import datetime
from importlib import import_module
from os import path
from time import mktime, strptime
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Generator, Iterable, List,
from urllib.parse import parse_qsl, quote_plus, urlencode, urlsplit, urlunsplit
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.errors import ExtensionError, FiletypeNotFoundError, SphinxParallelError
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.console import bold, colorize, strip_colors, term_width_line  # type: ignore
from sphinx.util.matching import patfilter  # noqa
from sphinx.util.nodes import (caption_ref_re, explicit_title_re,  # noqa
from sphinx.util.osutil import (SEP, copyfile, copytimes, ensuredir, make_filename,  # noqa
from sphinx.util.typing import PathMatcher
def split_index_msg(type: str, value: str) -> List[str]:
    if type == 'single':
        try:
            result = split_into(2, 'single', value)
        except ValueError:
            result = split_into(1, 'single', value)
    elif type == 'pair':
        result = split_into(2, 'pair', value)
    elif type == 'triple':
        result = split_into(3, 'triple', value)
    elif type == 'see':
        result = split_into(2, 'see', value)
    elif type == 'seealso':
        result = split_into(2, 'see', value)
    else:
        raise ValueError('invalid %s index entry %r' % (type, value))
    return result