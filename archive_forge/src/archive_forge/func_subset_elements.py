from __future__ import annotations
import re
from functools import lru_cache
from itertools import chain, count
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple
from fontTools import ttLib
from fontTools.subset.util import _add_method
from fontTools.ttLib.tables.S_V_G_ import SVGDocument
def subset_elements(el: etree.Element, retained_ids: Set[str]) -> bool:
    if el.attrib.get('id') in retained_ids:
        return True
    if any([subset_elements(e, retained_ids) for e in el]):
        return True
    assert len(el) == 0
    parent = el.getparent()
    if parent is not None:
        parent.remove(el)
    return False