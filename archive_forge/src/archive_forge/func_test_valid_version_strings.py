from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
def test_valid_version_strings(self):

    def _test_string(version, exp_major, exp_minor):
        v = api_versions.APIVersion(version)
        self.assertEqual(v.ver_major, exp_major)
        self.assertEqual(v.ver_minor, exp_minor)
    _test_string('1.1', 1, 1)
    _test_string('2.10', 2, 10)
    _test_string('5.234', 5, 234)
    _test_string('12.5', 12, 5)
    _test_string('2.0', 2, 0)
    _test_string('2.200', 2, 200)