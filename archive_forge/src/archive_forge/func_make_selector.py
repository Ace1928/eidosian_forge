import json
import typing
import warnings
from io import BytesIO
from typing import (
from warnings import warn
import jmespath
from lxml import etree, html
from packaging.version import Version
from .csstranslator import GenericTranslator, HTMLTranslator
from .utils import extract_regex, flatten, iflatten, shorten
def make_selector(x: Any) -> _SelectorType:
    if isinstance(x, str):
        return self.__class__(text=x, _expr=query, type='text')
    else:
        return self.__class__(root=x, _expr=query)