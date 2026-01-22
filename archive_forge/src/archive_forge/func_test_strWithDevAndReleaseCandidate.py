from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_strWithDevAndReleaseCandidate(self):
    """
        Calling C{str} on a version with a release candidate and dev release
        includes the release candidate and the dev release.
        """
    self.assertEqual(str(Version('dummy', 1, 0, 0, release_candidate=1, dev=2)), '[dummy, version 1.0.0.rc1.dev2]')