import datetime
import os
import shutil
import tempfile
import time
import types
import warnings
from dulwich.tests import SkipTest
from ..index import commit_tree
from ..objects import Commit, FixedSha, Tag, object_class
from ..pack import (
from ..repo import Repo
def make_object(cls, **attrs):
    """Make an object for testing and assign some members.

    This method creates a new subclass to allow arbitrary attribute
    reassignment, which is not otherwise possible with objects having
    __slots__.

    Args:
      attrs: dict of attributes to set on the new object.
    Returns: A newly initialized object of type cls.
    """

    class TestObject(cls):
        """Class that inherits from the given class, but without __slots__.

        Note that classes with __slots__ can't have arbitrary attributes
        monkey-patched in, so this is a class that is exactly the same only
        with a __dict__ instead of __slots__.
        """
    TestObject.__name__ = 'TestObject_' + cls.__name__
    obj = TestObject()
    for name, value in attrs.items():
        if name == 'id':
            sha = FixedSha(value)
            obj.sha = lambda: sha
        else:
            setattr(obj, name, value)
    return obj