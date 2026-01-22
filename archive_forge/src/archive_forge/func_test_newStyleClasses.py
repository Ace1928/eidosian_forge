import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_newStyleClasses(self):
    uj = jelly.unjelly(D)
    self.assertIs(D, uj)