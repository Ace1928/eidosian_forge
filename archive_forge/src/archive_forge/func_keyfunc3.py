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
def keyfunc3(item: Tuple[str, List]) -> str:
    k, v = item
    v[1] = sorted(((si, se) for si, (se, void, void) in v[1].items()), key=keyfunc2)
    if v[2] is None:
        if k.startswith('\u200f'):
            k = k[1:]
        letter = unicodedata.normalize('NFD', k[0])[0].upper()
        if letter.isalpha() or letter == '_':
            return letter
        else:
            return _('Symbols')
    else:
        return v[2]