import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_from_pip_string_legacy_non_440_beta(self):
    expected = version.SemanticVersion(2014, 2, prerelease_type='b', prerelease=2)
    parsed = from_pip_string('2014.2.b2')
    self.assertEqual(expected, parsed)