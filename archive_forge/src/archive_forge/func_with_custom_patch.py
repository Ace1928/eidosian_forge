import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def with_custom_patch(target):
    getter, attribute = _get_target(target)
    return custom_patch(getter, attribute, DEFAULT, None, False, None, None, None, {})