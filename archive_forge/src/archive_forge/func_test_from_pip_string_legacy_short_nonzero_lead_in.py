import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_from_pip_string_legacy_short_nonzero_lead_in(self):
    expected = version.SemanticVersion(0, 1, 0, prerelease_type='a', prerelease=2)
    parsed = from_pip_string('0.1a2')
    self.assertEqual(expected, parsed)