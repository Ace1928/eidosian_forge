import re
import pathlib
from itertools import chain
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit
from .exceptions import XMLSchemaValueError
from .resources import _PurePath, is_remote_url
from .translation import gettext as _
def replace_location(text: str, location: str, repl_location: str) -> str:
    repl = 'schemaLocation="{}"'.format(repl_location)
    pattern = '\\bschemaLocation\\s*=\\s*[\\\'\\"].*%s.*[\\\'"]' % re.escape(location)
    return re.sub(pattern, repl, text)