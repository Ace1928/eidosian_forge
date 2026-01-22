import ast
import hashlib
import inspect
import os
import re
import warnings
from collections import deque
from contextlib import contextmanager
from functools import partial
from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import (
from w3lib.html import replace_entities
from scrapy.item import Item
from scrapy.utils.datatypes import LocalWeakReferencedCache
from scrapy.utils.deprecate import ScrapyDeprecationWarning
from scrapy.utils.python import flatten, to_unicode
def extract_regex(regex: Union[str, Pattern], text: str, encoding: str='utf-8') -> List[str]:
    """Extract a list of unicode strings from the given text/encoding using the following policies:

    * if the regex contains a named group called "extract" that will be returned
    * if the regex contains multiple numbered groups, all those will be returned (flattened)
    * if the regex doesn't contain any group the entire regex matching is returned
    """
    warnings.warn('scrapy.utils.misc.extract_regex has moved to parsel.utils.extract_regex.', ScrapyDeprecationWarning, stacklevel=2)
    if isinstance(regex, str):
        regex = re.compile(regex, re.UNICODE)
    try:
        strings = [regex.search(text).group('extract')]
    except Exception:
        strings = regex.findall(text)
    strings = flatten(strings)
    if isinstance(text, str):
        return [replace_entities(s, keep=['lt', 'amp']) for s in strings]
    return [replace_entities(to_unicode(s, encoding), keep=['lt', 'amp']) for s in strings]