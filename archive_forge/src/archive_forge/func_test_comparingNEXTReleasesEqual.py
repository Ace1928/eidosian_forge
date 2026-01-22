from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_comparingNEXTReleasesEqual(self):
    """
        NEXT releases are equal to each other.
        """
    va = Version('whatever', 'NEXT', 0, 0)
    vb = Version('whatever', 'NEXT', 0, 0)
    self.assertEquals(vb, va)