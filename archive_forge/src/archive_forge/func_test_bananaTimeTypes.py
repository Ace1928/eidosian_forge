import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_bananaTimeTypes(self):
    """
        Jellying L{datetime.time}, L{datetime.timedelta}, L{datetime.datetime},
        and L{datetime.date} objects should result in jellied objects which can
        be serialized and unserialized with banana.
        """
    sampleDate = datetime.date(2020, 7, 11)
    sampleTime = datetime.time(1, 16, 5, 344)
    sampleDateTime = datetime.datetime.combine(sampleDate, sampleTime)
    sampleTimeDelta = sampleDateTime - datetime.datetime(2020, 7, 3)
    jellyRoundTrip(self, sampleDate)
    jellyRoundTrip(self, sampleTime)
    jellyRoundTrip(self, sampleDateTime)
    jellyRoundTrip(self, sampleTimeDelta)