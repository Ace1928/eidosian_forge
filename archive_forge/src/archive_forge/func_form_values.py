import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def form_values(self):
    """
        Return a list of tuples of the field values for the form.
        This is suitable to be passed to ``urllib.urlencode()``.
        """
    results = []
    for el in self.inputs:
        name = el.name
        if not name or 'disabled' in el.attrib:
            continue
        tag = _nons(el.tag)
        if tag == 'textarea':
            results.append((name, el.value))
        elif tag == 'select':
            value = el.value
            if el.multiple:
                for v in value:
                    results.append((name, v))
            elif value is not None:
                results.append((name, el.value))
        else:
            assert tag == 'input', 'Unexpected tag: %r' % el
            if el.checkable and (not el.checked):
                continue
            if el.type in ('submit', 'image', 'reset', 'file'):
                continue
            value = el.value
            if value is not None:
                results.append((name, el.value))
    return results