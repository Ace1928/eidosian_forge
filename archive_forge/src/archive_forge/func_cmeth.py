import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
@classmethod
def cmeth(cls, a, b, c, d=None):
    pass