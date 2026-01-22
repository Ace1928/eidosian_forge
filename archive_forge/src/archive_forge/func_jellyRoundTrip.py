import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def jellyRoundTrip(testCase, toSerialize):
    """
    Verify that the given object round-trips through jelly & banana and comes
    out equivalent to the input.
    """
    jellied = jelly.jelly(toSerialize)
    encoded = banana.encode(jellied)
    decoded = banana.decode(encoded)
    unjellied = jelly.unjelly(decoded)
    testCase.assertEqual(toSerialize, unjellied)