from oslo_serialization import jsonutils
from oslo_utils import timeutils
import uuid
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import orders
def test_create_asymmetric_order_w_legacy_pass_phrase_param(self):
    data = {'order_ref': self.entity_href}
    self.responses.post(self.entity_base + '/', json=data)
    passphrase = str(uuid.uuid4())
    order = orders.AsymmetricOrder(api=self.manager._api, name='name', algorithm='algorithm', payload_content_type='payload_content_type', pass_phrase=passphrase)
    order_href = order.submit()
    self.assertEqual(self.entity_href, order_href)
    self.assertEqual(passphrase, order.pass_phrase)