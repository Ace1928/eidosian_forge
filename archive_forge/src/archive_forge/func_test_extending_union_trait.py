import unittest
from traits.api import (
def test_extending_union_trait(self):

    class UnionAllowStr(Union):

        def validate(self, obj, name, value):
            if isinstance(value, str):
                return value
            return super().validate(obj, name, value)

    class TestClass(HasTraits):
        s = UnionAllowStr(Int, Float)
    TestClass(s='sdf')