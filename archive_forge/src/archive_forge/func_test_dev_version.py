import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_dev_version(self):
    semver = version.SemanticVersion(1, 2, 4, dev_count=5)
    self.assertEqual((1, 2, 4, 'dev', 4), semver.version_tuple())
    self.assertEqual('1.2.4', semver.brief_string())
    self.assertEqual('1.2.4~dev5', semver.debian_string())
    self.assertEqual('1.2.4.dev5', semver.release_string())
    self.assertEqual('1.2.3.dev5', semver.rpm_string())
    self.assertEqual(semver, from_pip_string('1.2.4.dev5'))