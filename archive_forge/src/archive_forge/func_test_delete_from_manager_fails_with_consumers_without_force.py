from barbicanclient import client
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_delete_secret_for_responses
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.tests.v1.test_secrets import SecretData
from barbicanclient.v1 import secrets
from oslo_serialization import jsonutils
def test_delete_from_manager_fails_with_consumers_without_force(self):
    self.assertRaises(ValueError, self._delete_from_manager_with_consumers, self.entity_href, force=False)