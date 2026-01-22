import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_split_path_failed(self):
    self.assertRaises(ValueError, strutils.split_path, '')
    self.assertRaises(ValueError, strutils.split_path, '/')
    self.assertRaises(ValueError, strutils.split_path, '//')
    self.assertRaises(ValueError, strutils.split_path, '//a')
    self.assertRaises(ValueError, strutils.split_path, '/a/c')
    self.assertRaises(ValueError, strutils.split_path, '//c')
    self.assertRaises(ValueError, strutils.split_path, '/a/c/')
    self.assertRaises(ValueError, strutils.split_path, '/a//')
    self.assertRaises(ValueError, strutils.split_path, '/a', 2)
    self.assertRaises(ValueError, strutils.split_path, '/a', 2, 3)
    self.assertRaises(ValueError, strutils.split_path, '/a', 2, 3, True)
    self.assertRaises(ValueError, strutils.split_path, '/a/c/o/r', 3, 3)
    self.assertRaises(ValueError, strutils.split_path, '/a', 5, 4)