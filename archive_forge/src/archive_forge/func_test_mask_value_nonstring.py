import uuid
from pycadf.tests import base
from pycadf import utils
def test_mask_value_nonstring(self):
    value = 12
    obfuscate = utils.mask_value(value)
    self.assertEqual(value, obfuscate)