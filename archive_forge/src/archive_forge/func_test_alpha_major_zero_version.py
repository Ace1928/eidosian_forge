import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_alpha_major_zero_version(self):
    semver = version.SemanticVersion(1, 0, 0, 'a', 1)
    self.assertEqual((1, 0, 0, 'alpha', 1), semver.version_tuple())
    self.assertEqual('1.0.0', semver.brief_string())
    self.assertEqual('1.0.0~a1', semver.debian_string())
    self.assertEqual('1.0.0.0a1', semver.release_string())
    self.assertEqual('0.9999.9999.a1', semver.rpm_string())
    self.assertEqual(semver, from_pip_string('1.0.0.0a1'))