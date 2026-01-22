import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_ensure_binary_raise_type_error(self):
    with pytest.raises(TypeError):
        six.ensure_str(8)