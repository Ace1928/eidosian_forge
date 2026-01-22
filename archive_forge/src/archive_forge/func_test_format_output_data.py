from unittest import mock
import testtools
from blazarclient import command
from blazarclient import tests
def test_format_output_data(self):
    data_before = {'key_string': 'string_value', 'key_dict': {'key': 'value'}, 'key_list': ['1', '2', '3'], 'key_none': None}
    data_after = {'key_string': 'string_value', 'key_dict': '{"key": "value"}', 'key_list': '1\n2\n3', 'key_none': ''}
    self.command.format_output_data(data_before)
    self.assertEqual(data_after, data_before)