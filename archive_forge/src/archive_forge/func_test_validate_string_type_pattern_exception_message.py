import re
import unittest
from wsme import exc
from wsme import types
def test_validate_string_type_pattern_exception_message(self):
    regex = '^[a-zA-Z0-9]*$'
    v = types.StringType(pattern=regex)
    try:
        v.validate('_')
        self.assertFail()
    except ValueError as e:
        self.assertIn(regex, str(e))