from __future__ import annotations
import re
from functools import lru_cache
from itertools import chain, count
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple
from fontTools import ttLib
from fontTools.subset.util import _add_method
from fontTools.ttLib.tables.S_V_G_ import SVGDocument
def parse_css_declarations(style_attr: str) -> Dict[str, str]:
    result = {}
    for declaration in style_attr.split(';'):
        if declaration.count(':') == 1:
            property_name, value = declaration.split(':')
            property_name = property_name.strip()
            result[property_name] = value.strip()
        elif declaration.strip():
            raise ValueError(f'Invalid CSS declaration syntax: {declaration}')
    return result