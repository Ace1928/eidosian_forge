import copy
from osc_lib.api import api
from osc_lib.api import utils as api_utils
from osc_lib.tests.api import fakes as api_fakes
def test_simple_filter_attr_value(self):
    output = api_utils.simple_filter(copy.deepcopy(self.input_list), attr='status', value='')
    self.assertEqual([], output)
    output = api_utils.simple_filter(copy.deepcopy(self.input_list), attr='status', value='UP')
    self.assertEqual([api_fakes.RESP_ITEM_1, api_fakes.RESP_ITEM_3], output)
    output = api_utils.simple_filter(copy.deepcopy(self.input_list), attr='fred', value='UP')
    self.assertEqual([], output)