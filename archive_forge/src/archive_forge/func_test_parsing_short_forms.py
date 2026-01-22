import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_parsing_short_forms(self):
    semver = version.SemanticVersion(1, 0, 0)
    self.assertEqual(semver, from_pip_string('1'))
    self.assertEqual(semver, from_pip_string('1.0'))
    self.assertEqual(semver, from_pip_string('1.0.0'))