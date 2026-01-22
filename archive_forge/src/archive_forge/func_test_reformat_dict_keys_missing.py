from heat.api.aws import exception as aws_exception
from heat.api.aws import utils as api_utils
from heat.common import exception as common_exception
from heat.tests import common
def test_reformat_dict_keys_missing(self):
    keymap = {'foo': 'bar', 'foo2': 'bar2'}
    data = {'foo': 123}
    expected = {'bar': 123}
    result = api_utils.reformat_dict_keys(keymap, data)
    self.assertEqual(expected, result)