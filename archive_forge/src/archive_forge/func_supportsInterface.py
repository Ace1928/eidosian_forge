import sys
from typing import Optional, Sequence, Type
from zope.interface import Attribute, Interface
from twisted.plugin import getPlugins
from twisted.python import usage
def supportsInterface(self, interface):
    """
        Returns whether a particular credentials interface is supported.
        """
    return self.supportedInterfaces is None or interface in self.supportedInterfaces