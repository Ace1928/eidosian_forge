from __future__ import absolute_import, division, unicode_literals
from xml.dom import minidom, Node
import weakref
from . import base
from .. import constants
from ..constants import namespaces
from .._utils import moduleFactoryFactory
def setAttributes(self, attributes):
    if attributes:
        for name, value in list(attributes.items()):
            if isinstance(name, tuple):
                if name[0] is not None:
                    qualifiedName = name[0] + ':' + name[1]
                else:
                    qualifiedName = name[1]
                self.element.setAttributeNS(name[2], qualifiedName, value)
            else:
                self.element.setAttribute(name, value)