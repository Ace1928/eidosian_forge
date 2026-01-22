import re
import unittest
from oslo_config import types
def test_concrete_class(self):

    class MyString(types.ConfigType):

        def __init__(self, type_name='mystring value'):
            super(MyString, self).__init__(type_name=type_name)

        def _formatter(self, value):
            return value
    MyString()