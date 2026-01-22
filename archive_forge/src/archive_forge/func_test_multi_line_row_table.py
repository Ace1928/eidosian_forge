import ddt
from manilaclient.tests.functional import utils as func_utils
from manilaclient.tests.unit import utils
@ddt.data({'input': OUTPUT_LINES_SIMPLE, 'valid_values': [['11', 'foo', 'BUILD'], ['21', 'bar', 'ERROR']]}, {'input': OUTPUT_LINES_ONE_MULTI_ROW, 'valid_values': [['11', 'foo', 'BUILD'], ['21', 'bar', ['ERROR', 'ERROR2']], ['31', 'bee', 'None']]}, {'input': OUTPUT_LINES_COMPLICATED_MULTI_ROW, 'valid_values': [['11', 'foo', 'BUILD'], ['21', 'bar', ['ERROR', 'ERROR2', 'ERROR3']], ['31', ['bee', 'bee2', 'bee3'], 'None'], ['41', ['rand', 'rend'], ['None', 'None2']], ['', '', '']]})
@ddt.unpack
def test_multi_line_row_table(self, input, valid_values):
    actual_result = func_utils.multi_line_row_table(input)
    self.assertEqual(['ID', 'Name', 'Status'], actual_result['headers'])
    self.assertEqual(valid_values, actual_result['values'])