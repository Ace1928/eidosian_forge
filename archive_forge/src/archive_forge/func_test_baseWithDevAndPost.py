from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_baseWithDevAndPost(self):
    """
        The base version includes 'postXdevX' for versions with dev releases
        and a postrelease.
        """
    self.assertEqual(Version('foo', 1, 0, 0, post=2, dev=8).base(), '1.0.0.post2.dev8')