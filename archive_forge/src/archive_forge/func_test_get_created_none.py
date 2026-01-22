from castellan.common import objects
from castellan.common.objects import private_key
from castellan.tests import base
from castellan.tests import utils
def test_get_created_none(self):
    created = None
    key = private_key.PrivateKey(self.algorithm, self.bit_length, self.encoded, self.name, created, consumers=self.consumers)
    self.assertEqual(created, key.created)