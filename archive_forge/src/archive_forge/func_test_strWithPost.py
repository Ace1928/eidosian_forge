from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_strWithPost(self):
    """
        Calling C{str} on a version with a postrelease includes the
        postrelease.
        """
    self.assertEqual(str(Version('dummy', 1, 0, 0, post=1)), '[dummy, version 1.0.0.post1]')