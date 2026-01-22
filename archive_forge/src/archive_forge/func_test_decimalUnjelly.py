import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_decimalUnjelly(self):
    """
        Unjellying the s-expressions produced by jelly for L{decimal.Decimal}
        instances should result in L{decimal.Decimal} instances with the values
        represented by the s-expressions.

        This test also verifies that L{decimalData} contains valid jellied
        data.  This is important since L{test_decimalMissing} re-uses
        L{decimalData} and is expected to be unable to produce
        L{decimal.Decimal} instances even though the s-expression correctly
        represents a list of them.
        """
    expected = [decimal.Decimal('9.95'), decimal.Decimal(0), decimal.Decimal(123456), decimal.Decimal('-78.901')]
    output = jelly.unjelly(self.decimalData)
    self.assertEqual(output, expected)