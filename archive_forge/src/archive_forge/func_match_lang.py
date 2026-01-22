from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_lang(self, el: bs4.Tag, langs: tuple[ct.SelectorLang, ...]) -> bool:
    """Match languages."""
    match = False
    has_ns = self.supports_namespaces()
    root = self.root
    has_html_namespace = self.has_html_namespace
    parent = el
    found_lang = None
    last = None
    while not found_lang:
        has_html_ns = self.has_html_ns(parent)
        for k, v in self.iter_attributes(parent):
            attr_ns, attr = self.split_namespace(parent, k)
            if (not has_ns or has_html_ns) and (util.lower(k) if not self.is_xml else k) == 'lang' or (has_ns and (not has_html_ns) and (attr_ns == NS_XML) and ((util.lower(attr) if not self.is_xml and attr is not None else attr) == 'lang')):
                found_lang = v
                break
        last = parent
        parent = self.get_parent(parent, no_iframe=self.is_html)
        if parent is None:
            root = last
            has_html_namespace = self.has_html_ns(root)
            parent = last
            break
    if found_lang is None and self.cached_meta_lang:
        for cache in self.cached_meta_lang:
            if root is cache[0]:
                found_lang = cache[1]
    if found_lang is None and (not self.is_xml or (has_html_namespace and root.name == 'html')):
        found = False
        for tag in ('html', 'head'):
            found = False
            for child in self.get_children(parent, no_iframe=self.is_html):
                if self.get_tag(child) == tag and self.is_html_tag(child):
                    found = True
                    parent = child
                    break
            if not found:
                break
        if found:
            for child in parent:
                if self.is_tag(child) and self.get_tag(child) == 'meta' and self.is_html_tag(parent):
                    c_lang = False
                    content = None
                    for k, v in self.iter_attributes(child):
                        if util.lower(k) == 'http-equiv' and util.lower(v) == 'content-language':
                            c_lang = True
                        if util.lower(k) == 'content':
                            content = v
                        if c_lang and content:
                            found_lang = content
                            self.cached_meta_lang.append((cast(str, root), cast(str, found_lang)))
                            break
                if found_lang is not None:
                    break
            if found_lang is None:
                self.cached_meta_lang.append((cast(str, root), ''))
    if found_lang is not None:
        for patterns in langs:
            match = False
            for pattern in patterns:
                if self.extended_language_filter(pattern, cast(str, found_lang)):
                    match = True
            if not match:
                break
    return match