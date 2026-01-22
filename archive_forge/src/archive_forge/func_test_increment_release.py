import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_increment_release(self):
    semver = version.SemanticVersion(1, 2, 5)
    self.assertEqual(version.SemanticVersion(1, 2, 6), semver.increment())
    self.assertEqual(version.SemanticVersion(1, 3, 0), semver.increment(minor=True))
    self.assertEqual(version.SemanticVersion(2, 0, 0), semver.increment(major=True))