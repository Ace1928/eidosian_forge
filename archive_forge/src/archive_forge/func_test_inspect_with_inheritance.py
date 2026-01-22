import re
import unittest
from wsme import exc
from wsme import types
def test_inspect_with_inheritance(self):

    class Parent(object):
        parent_attribute = int

    class Child(Parent):
        child_attribute = int
    types.register_type(Parent)
    types.register_type(Child)
    assert len(Child._wsme_attributes) == 2