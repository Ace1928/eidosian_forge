from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_baseWithPost(self):
    """
        The base version includes 'postX' for versions with postreleases.
        """
    self.assertEqual(Version('foo', 1, 0, 0, post=8).base(), '1.0.0.post8')