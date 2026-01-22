from openstackclient.tests.functional.identity.v2 import common
def test_endpoint_multi_delete(self):
    endpoint_id_1 = self._create_dummy_endpoint(add_clean_up=False)
    endpoint_id_2 = self._create_dummy_endpoint(add_clean_up=False)
    raw_output = self.openstack('endpoint delete ' + endpoint_id_1 + ' ' + endpoint_id_2)
    self.assertEqual(0, len(raw_output))