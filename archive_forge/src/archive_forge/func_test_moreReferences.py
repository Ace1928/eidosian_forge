import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_moreReferences(self):
    a = []
    t = (a,)
    a.append((t,))
    s = jelly.jelly(t)
    z = jelly.unjelly(s)
    self.assertIs(z[0][0][0], z)