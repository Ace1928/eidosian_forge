from oslo_utils import timeutils
from barbicanclient.tests import test_client
from barbicanclient.v1 import cas
def test_should_get_list(self):
    ca_resp = self.entity_href
    data = {'cas': [ca_resp for v in range(3)]}
    m = self.responses.get(self.entity_base, json=data)
    ca_list = self.manager.list(limit=10, offset=5)
    self.assertTrue(len(ca_list) == 3)
    self.assertIsInstance(ca_list[0], cas.CA)
    self.assertEqual(self.entity_href, ca_list[0].ca_ref)
    self.assertEqual(self.entity_base, m.last_request.url.split('?')[0])
    self.assertEqual(['10'], m.last_request.qs['limit'])
    self.assertEqual(['5'], m.last_request.qs['offset'])