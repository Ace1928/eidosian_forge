from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_NEXTMustBeAlone(self):
    """
        NEXT releases must always have the rest of the numbers set to 0.
        """
    with self.assertRaises(ValueError):
        Version('whatever', 'NEXT', 1, 0, release_candidate=0, post=0, dev=0)
    with self.assertRaises(ValueError):
        Version('whatever', 'NEXT', 0, 1, release_candidate=0, post=0, dev=0)
    with self.assertRaises(ValueError):
        Version('whatever', 'NEXT', 0, 0, release_candidate=1, post=0, dev=0)
    with self.assertRaises(ValueError):
        Version('whatever', 'NEXT', 0, 0, release_candidate=0, post=1, dev=0)
    with self.assertRaises(ValueError):
        Version('whatever', 'NEXT', 0, 0, release_candidate=0, post=0, dev=1)