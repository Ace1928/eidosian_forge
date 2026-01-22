from castellan.common import objects
from castellan.common.objects import private_key
from castellan.tests import base
from castellan.tests import utils
def test___ne___consumers(self):
    different_consumers = [{'service': 'other_service', 'resource_type': 'other_type', 'resource_id': 'other_id'}]
    other_key = private_key.PrivateKey(self.algorithm, self.bit_length, self.encoded, self.name, consumers=different_consumers)
    self.assertTrue(self.key is not other_key)