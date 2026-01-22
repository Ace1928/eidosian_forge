import re
import unicodedata
from itertools import groupby
from typing import Any, Dict, List, Optional, Pattern, Tuple, cast
from sphinx.builders import Builder
from sphinx.domains.index import IndexDomain
from sphinx.environment import BuildEnvironment
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.util import logging, split_into
def keyfunc2(entry: Tuple[str, List]) -> str:
    key = unicodedata.normalize('NFD', entry[0].lower())
    if key.startswith('\u200f'):
        key = key[1:]
    if key[0:1].isalpha() or key.startswith('_'):
        key = chr(127) + key
    return key