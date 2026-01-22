from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_default(self, el: bs4.Tag) -> bool:
    """Match default."""
    match = False
    form = None
    parent = self.get_parent(el, no_iframe=True)
    while parent and form is None:
        if self.get_tag(parent) == 'form' and self.is_html_tag(parent):
            form = parent
        else:
            parent = self.get_parent(parent, no_iframe=True)
    found_form = False
    for f, t in self.cached_default_forms:
        if f is form:
            found_form = True
            if t is el:
                match = True
            break
    if not found_form:
        for child in self.get_descendants(form, no_iframe=True):
            name = self.get_tag(child)
            if name == 'form':
                break
            if name in ('input', 'button'):
                v = self.get_attribute_by_name(child, 'type', '')
                if v and util.lower(v) == 'submit':
                    self.cached_default_forms.append((form, child))
                    if el is child:
                        match = True
                    break
    return match