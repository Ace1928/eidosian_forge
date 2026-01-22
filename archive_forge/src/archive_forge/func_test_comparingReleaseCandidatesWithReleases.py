from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_comparingReleaseCandidatesWithReleases(self):
    """
        Release Candidates are always less than versions without release
        candidates.
        """
    va = Version('whatever', 1, 0, 0, release_candidate=1)
    vb = Version('whatever', 1, 0, 0)
    self.assertTrue(va < vb)
    self.assertFalse(va > vb)
    self.assertNotEquals(vb, va)