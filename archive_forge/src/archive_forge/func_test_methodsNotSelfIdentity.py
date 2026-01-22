import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_methodsNotSelfIdentity(self):
    """
        If a class change after an instance has been created, L{jelly.unjelly}
        shoud raise a C{TypeError} when trying to unjelly the instance.
        """
    a = A()
    b = B()
    c = C()
    a.bmethod = c.cmethod
    b.a = a
    savecmethod = C.cmethod
    del C.cmethod
    try:
        self.assertRaises(TypeError, jelly.unjelly, jelly.jelly(b))
    finally:
        C.cmethod = savecmethod