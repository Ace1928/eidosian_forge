import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
def test_property_notifications(self):
    output_buffer = io.StringIO()
    test_obj = Test_1(output_buffer=output_buffer)
    test_obj.value = 'value_1'
    self.assertEqual(output_buffer.getvalue(), 'value_1')
    test_obj.value = 'value_2'
    self.assertEqual(output_buffer.getvalue(), 'value_1value_2')