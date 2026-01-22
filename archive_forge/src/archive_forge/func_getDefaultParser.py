from __future__ import print_function, absolute_import
import threading
import warnings
from lxml import etree as _etree
from .common import DTDForbidden, EntitiesForbidden, NotSupportedError
def getDefaultParser(self):
    parser = getattr(self, '_default_parser', None)
    if parser is None:
        parser = self.createDefaultParser()
        self.setDefaultParser(parser)
    return parser