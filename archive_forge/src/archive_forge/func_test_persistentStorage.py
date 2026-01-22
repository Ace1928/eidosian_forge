import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_persistentStorage(self):
    perst = [{}, 1]

    def persistentStore(obj, jel, perst=perst):
        perst[1] = perst[1] + 1
        perst[0][perst[1]] = obj
        return str(perst[1])

    def persistentLoad(pidstr, unj, perst=perst):
        pid = int(pidstr)
        return perst[0][pid]
    a = SimpleJellyTest(1, 2)
    b = SimpleJellyTest(3, 4)
    c = SimpleJellyTest(5, 6)
    a.b = b
    a.c = c
    c.b = b
    jel = jelly.jelly(a, persistentStore=persistentStore)
    x = jelly.unjelly(jel, persistentLoad=persistentLoad)
    self.assertIs(x.b, x.c.b)
    self.assertTrue(perst[0], 'persistentStore was not called.')
    self.assertIs(x.b, a.b, 'Persistent storage identity failure.')