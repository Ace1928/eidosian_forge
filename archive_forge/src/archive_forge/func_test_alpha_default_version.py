import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_alpha_default_version(self):
    semver = version.SemanticVersion(1, 2, 4, 'a')
    self.assertEqual((1, 2, 4, 'alpha', 0), semver.version_tuple())
    self.assertEqual('1.2.4', semver.brief_string())
    self.assertEqual('1.2.4~a0', semver.debian_string())
    self.assertEqual('1.2.4.0a0', semver.release_string())
    self.assertEqual('1.2.3.a0', semver.rpm_string())
    self.assertEqual(semver, from_pip_string('1.2.4.0a0'))