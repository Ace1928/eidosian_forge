import json
import testtools
from troveclient.v1 import metadata
from unittest import mock
def test_parse_value_invalid_json_in(self):
    value = "{'one': [2, 3, 4]}"
    new_value = self.metadata._parse_value(value)
    self.assertEqual(value, new_value)