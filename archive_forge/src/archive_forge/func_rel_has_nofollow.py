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
def rel_has_nofollow(rel: Optional[str]) -> bool:
    """Return True if link rel attribute has nofollow type"""
    return rel is not None and 'nofollow' in rel.replace(',', ' ').split()