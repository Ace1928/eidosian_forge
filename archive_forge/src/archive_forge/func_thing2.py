import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
@patch.object(Foo, 'foo', new_callable=crasher)
@patch.object(Foo, 'g', 1)
@patch.object(Foo, 'f', 1)
def thing2():
    pass