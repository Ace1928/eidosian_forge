import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_from_pip_string_legacy_alpha(self):
    expected = version.SemanticVersion(1, 2, 0, prerelease_type='rc', prerelease=1)
    parsed = from_pip_string('1.2.0rc1')
    self.assertEqual(expected, parsed)