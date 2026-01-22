from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
def test_get_element_associated_class_specific_fields(self):
    self._test_get_element_associated_class(fields=['field', 'another_field'])