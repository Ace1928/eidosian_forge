from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_baseWithReleaseCandidate(self):
    """
        The base version includes 'rcX' for versions with prereleases.
        """
    self.assertEqual(Version('foo', 1, 0, 0, release_candidate=8).base(), '1.0.0.rc8')