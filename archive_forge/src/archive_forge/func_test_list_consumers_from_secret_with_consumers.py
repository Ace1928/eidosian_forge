from barbicanclient import client
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_delete_secret_for_responses
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.tests.v1.test_secrets import SecretData
from barbicanclient.v1 import secrets
from oslo_serialization import jsonutils
def test_list_consumers_from_secret_with_consumers(self):
    consumers = [{'service': 'service_test1', 'resource_type': 'type_test1', 'resource_id': 'id_test1'}, {'service': 'service_test2', 'resource_type': 'type_test2', 'resource_id': 'id_test2'}]
    consumer_list = self._list_consumers(self.entity_href, consumers)
    for elem in range(len(consumers)):
        self.assertTrue(consumer_list[elem].service == consumers[elem]['service'])
        self.assertTrue(consumer_list[elem].resource_type == consumers[elem]['resource_type'])
        self.assertTrue(consumer_list[elem].resource_id == consumers[elem]['resource_id'])