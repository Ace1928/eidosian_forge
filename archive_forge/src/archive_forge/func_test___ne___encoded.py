from castellan.common import objects
from castellan.common.objects import private_key
from castellan.tests import base
from castellan.tests import utils
def test___ne___encoded(self):
    different_encoded = bytes(utils.get_private_key_der()) + b'\x00'
    other_key = private_key.PrivateKey(self.algorithm, self.bit_length, different_encoded, self.name, consumers=self.consumers)
    self.assertTrue(self.key != other_key)