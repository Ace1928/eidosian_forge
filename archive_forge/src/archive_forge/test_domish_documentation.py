from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish

        Test that prefixes passed to serialization are not modified.

        This test makes sure that passing a dictionary of prefixes repeatedly
        to C{toXml} of elements does not cause serialization errors. A
        previous implementation changed the passed in dictionary internally,
        causing havoc later on.
        