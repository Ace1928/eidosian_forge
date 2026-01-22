from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
def test_get_available_client_versions(self):
    output = api_versions.get_available_major_versions()
    self.assertNotEqual([], output)