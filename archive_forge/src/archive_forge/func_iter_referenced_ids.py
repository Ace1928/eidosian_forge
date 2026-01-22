from __future__ import annotations
import re
from functools import lru_cache
from itertools import chain, count
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple
from fontTools import ttLib
from fontTools.subset.util import _add_method
from fontTools.ttLib.tables.S_V_G_ import SVGDocument
def iter_referenced_ids(tree: etree.Element) -> Iterator[str]:
    find_svg_elements_with_references = xpath(".//svg:*[ starts-with(@xlink:href, '#') or starts-with(@fill, 'url(#') or starts-with(@clip-path, 'url(#') or contains(@style, ':url(#') ]")
    for el in chain([tree], find_svg_elements_with_references(tree)):
        ref_id = href_local_target(el)
        if ref_id is not None:
            yield ref_id
        attrs = el.attrib
        if 'style' in attrs:
            attrs = {**dict(attrs), **parse_css_declarations(el.attrib['style'])}
        for attr in ('fill', 'clip-path'):
            if attr in attrs:
                value = attrs[attr]
                if value.startswith('url(#') and value.endswith(')'):
                    ref_id = value[5:-1]
                    assert ref_id
                    yield ref_id