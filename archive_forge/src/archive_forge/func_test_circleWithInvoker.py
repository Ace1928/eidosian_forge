import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_circleWithInvoker(self):

    class DummyInvokerClass:
        pass
    dummyInvoker = DummyInvokerClass()
    dummyInvoker.serializingPerspective = None
    a0 = ClassA()
    jelly.setUnjellyableForClass(ClassA, ClassA)
    jelly.setUnjellyableForClass(ClassB, ClassB)
    j = jelly.jelly(a0, invoker=dummyInvoker)
    a1 = jelly.unjelly(j)
    self.failUnlessIdentical(a1.ref.ref, a1, 'Identity not preserved in circular reference')