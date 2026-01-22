from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_past_relations(self, el: bs4.Tag, relation: ct.SelectorList) -> bool:
    """Match past relationship."""
    found = False
    if isinstance(relation[0], ct.SelectorNull):
        return found
    if relation[0].rel_type == REL_PARENT:
        parent = self.get_parent(el, no_iframe=self.iframe_restrict)
        while not found and parent:
            found = self.match_selectors(parent, relation)
            parent = self.get_parent(parent, no_iframe=self.iframe_restrict)
    elif relation[0].rel_type == REL_CLOSE_PARENT:
        parent = self.get_parent(el, no_iframe=self.iframe_restrict)
        if parent:
            found = self.match_selectors(parent, relation)
    elif relation[0].rel_type == REL_SIBLING:
        sibling = self.get_previous(el)
        while not found and sibling:
            found = self.match_selectors(sibling, relation)
            sibling = self.get_previous(sibling)
    elif relation[0].rel_type == REL_CLOSE_SIBLING:
        sibling = self.get_previous(el)
        if sibling and self.is_tag(sibling):
            found = self.match_selectors(sibling, relation)
    return found