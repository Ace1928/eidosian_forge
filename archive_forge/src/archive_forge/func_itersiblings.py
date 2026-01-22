from __future__ import print_function, absolute_import
import threading
import warnings
from lxml import etree as _etree
from .common import DTDForbidden, EntitiesForbidden, NotSupportedError
def itersiblings(self, tag=None, preceding=False):
    iterator = super(RestrictedElement, self).itersiblings(tag=tag, preceding=preceding)
    return self._filter(iterator)