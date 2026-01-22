from __future__ import absolute_import, division, unicode_literals
from xml.dom import minidom, Node
import weakref
from . import base
from .. import constants
from ..constants import namespaces
from .._utils import moduleFactoryFactory
def getNameTuple(self):
    if self.namespace is None:
        return (namespaces['html'], self.name)
    else:
        return (self.namespace, self.name)