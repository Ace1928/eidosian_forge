import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_decimal(self):
    """
        Jellying L{decimal.Decimal} instances and then unjellying the result
        should produce objects which represent the values of the original
        inputs.
        """
    inputList = [decimal.Decimal('9.95'), decimal.Decimal(0), decimal.Decimal(123456), decimal.Decimal('-78.901')]
    c = jelly.jelly(inputList)
    output = jelly.unjelly(c)
    self.assertEqual(inputList, output)
    self.assertIsNot(inputList, output)