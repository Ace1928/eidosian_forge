from __future__ import annotations
from . import Extension
from ..treeprocessors import Treeprocessor
from ..util import parseBoolValue, AMP_SUBSTITUTE, deprecated, HTML_PLACEHOLDER_RE, AtomicString
from ..treeprocessors import UnescapeTreeprocessor
from ..serializers import RE_AMP
import re
import html
import unicodedata
from copy import deepcopy
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any, Iterator, MutableSet
def remove_fnrefs(root: etree.Element) -> etree.Element:
    """ Remove footnote references from a copy of the element, if any are present. """
    if next(root.iter('sup'), None) is None:
        return root
    root = deepcopy(root)
    for parent in root.findall('.//sup/..'):
        carry_text = ''
        for child in reversed(parent):
            if child.tag == 'sup' and child.get('id', '').startswith('fnref'):
                carry_text = f'{child.tail or ''}{carry_text}'
                parent.remove(child)
            elif carry_text:
                child.tail = f'{child.tail or ''}{carry_text}'
                carry_text = ''
        if carry_text:
            parent.text = f'{parent.text or ''}{carry_text}'
    return root