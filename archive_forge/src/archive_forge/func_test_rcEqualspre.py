from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_rcEqualspre(self):
    """
        Release Candidates are equal to prereleases.
        """
    va = Version('whatever', 1, 0, 0, release_candidate=1)
    vb = Version('whatever', 1, 0, 0, prerelease=1)
    self.assertTrue(va == vb)
    self.assertFalse(va != vb)