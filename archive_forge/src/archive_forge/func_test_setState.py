import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_setState(self):
    global TupleState

    class TupleState:

        def __init__(self, other):
            self.other = other

        def __getstate__(self):
            return (self.other,)

        def __setstate__(self, state):
            self.other = state[0]

        def __hash__(self):
            return hash(self.other)
    a = A()
    t1 = TupleState(a)
    t2 = TupleState(a)
    t3 = TupleState((t1, t2))
    d = {t1: t1, t2: t2, t3: t3, 't3': t3}
    t3prime = jelly.unjelly(jelly.jelly(d))['t3']
    self.assertIs(t3prime.other[0].other, t3prime.other[1].other)