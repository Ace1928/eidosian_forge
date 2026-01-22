import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
def getTypeToCopy(self):
    """Determine what type tag to send for me.

        By default, send the string representation of my class
        (package.module.Class); normally this is adequate, but
        you may override this to change it.
        """
    return reflect.qual(self.__class__).encode('utf-8')