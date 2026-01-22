import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def toggle(self, value):
    """
        Add a class name if it isn't there yet, or remove it if it exists.

        Returns true if the class was added (and is now enabled) and
        false if it was removed (and is now disabled).
        """
    if not value or re.search('\\s', value):
        raise ValueError('Invalid class name: %r' % value)
    classes = self._get_class_value().split()
    try:
        classes.remove(value)
        enabled = False
    except ValueError:
        classes.append(value)
        enabled = True
    if classes:
        self._attributes['class'] = ' '.join(classes)
    else:
        del self._attributes['class']
    return enabled