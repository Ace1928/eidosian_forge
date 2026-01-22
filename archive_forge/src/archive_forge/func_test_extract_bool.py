from heat.common import param_utils
from heat.tests import common
def test_extract_bool(self):
    for value in ('True', 'true', 'TRUE', True):
        self.assertTrue(param_utils.extract_bool('bool', value))
    for value in ('False', 'false', 'FALSE', False):
        self.assertFalse(param_utils.extract_bool('bool', value))
    for value in ('foo', 't', 'f', 'yes', 'no', 'y', 'n', '1', '0', None):
        self.assertRaises(ValueError, param_utils.extract_bool, 'bool', value)