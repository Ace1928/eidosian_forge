import unittest
from traits.api import (
def test_default_raise_error(self):
    with self.assertRaises(ValueError) as exception_context:
        Union(Int(), Float(), default=1.0)
    self.assertEqual(str(exception_context.exception), "Union default value should be set via 'default_value', not 'default'.")