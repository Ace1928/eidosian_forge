from castellan.common import objects
from castellan.common.objects import private_key
from castellan.tests import base
from castellan.tests import utils
def test___ne___none(self):
    self.assertTrue(self.key is not None)
    self.assertTrue(None != self.key)