from barbicanclient import client
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_delete_secret_for_responses
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.tests.v1.test_secrets import SecretData
from barbicanclient.v1 import secrets
from oslo_serialization import jsonutils
def test_register_consumer_fails_with_lower_microversion(self):
    self.assertRaises(NotImplementedError, self.manager_v1_0.register_consumer, self.entity_href, self.secret.consumer.get('service'), self.secret.consumer.get('resource_type'), self.secret.consumer.get('resource_id'))