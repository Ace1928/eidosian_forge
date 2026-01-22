from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_comparingDev(self):
    """
        The value specified as the dev release is used in version comparisons.
        """
    va = Version('whatever', 1, 0, 0, dev=1)
    vb = Version('whatever', 1, 0, 0, dev=2)
    self.assertTrue(va < vb)
    self.assertTrue(vb > va)
    self.assertTrue(va <= vb)
    self.assertTrue(vb >= va)
    self.assertTrue(va != vb)
    self.assertTrue(vb == Version('whatever', 1, 0, 0, dev=2))
    self.assertTrue(va == va)