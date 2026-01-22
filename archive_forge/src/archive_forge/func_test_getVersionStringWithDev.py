from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_getVersionStringWithDev(self):
    """
        L{getVersionString} includes the dev release, if any.
        """
    self.assertEqual(getVersionString(Version('whatever', 8, 0, 0, dev=1)), 'whatever 8.0.0.dev1')