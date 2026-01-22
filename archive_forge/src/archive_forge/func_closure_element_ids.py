from __future__ import annotations
import re
from functools import lru_cache
from itertools import chain, count
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple
from fontTools import ttLib
from fontTools.subset.util import _add_method
from fontTools.ttLib.tables.S_V_G_ import SVGDocument
def closure_element_ids(elements: Dict[str, etree.Element], element_ids: Set[str]) -> None:
    unvisited = element_ids
    while unvisited:
        referenced: Set[str] = set()
        for el_id in unvisited:
            if el_id not in elements:
                continue
            referenced.update(iter_referenced_ids(elements[el_id]))
        referenced -= element_ids
        element_ids.update(referenced)
        unvisited = referenced