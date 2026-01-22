import random
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred.credentials import (
from twisted.cred.portal import Portal
from twisted.internet import defer, protocol
from twisted.persisted import styles
from twisted.python import failure, log, reflect
from twisted.python.compat import cmp, comparable
from twisted.python.components import registerAdapter
from twisted.spread import banana
from twisted.spread.flavors import (
from twisted.spread.interfaces import IJellyable, IUnjellyable
from twisted.spread.jelly import _newInstance, globalSecurity, jelly, unjelly
def proto_uncache(self, objectID):
    """
        (internal) Tell the client it is now OK to uncache an object.

        @param objectID: The object ID.
        """
    obj = self.locallyCachedObjects[objectID]
    obj.broker = None
    del self.locallyCachedObjects[objectID]