from __future__ import absolute_import, division, unicode_literals
from xml.dom import minidom, Node
import weakref
from . import base
from .. import constants
from ..constants import namespaces
from .._utils import moduleFactoryFactory
def elementClass(self, name, namespace=None):
    if namespace is None and self.defaultNamespace is None:
        node = self.dom.createElement(name)
    else:
        node = self.dom.createElementNS(namespace, name)
    return NodeBuilder(node)