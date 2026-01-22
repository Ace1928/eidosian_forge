import argparse
import enum
import os
import os.path
import pickle
import re
import sys
import types
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.config import (
from pyomo.common.log import LoggingIntercept
def test_In(self):
    c = ConfigDict()
    c.declare('a', ConfigValue(None, In([1, 3, 5])))
    self.assertEqual(c.get('a').domain_name(), 'In[1, 3, 5]')
    self.assertEqual(c.a, None)
    c.a = 3
    self.assertEqual(c.a, 3)
    with self.assertRaises(ValueError):
        c.a = 2
    self.assertEqual(c.a, 3)
    with self.assertRaises(ValueError):
        c.a = {}
    self.assertEqual(c.a, 3)
    with self.assertRaises(ValueError):
        c.a = '1'
    self.assertEqual(c.a, 3)
    c.declare('b', ConfigValue(None, In([1, 3, 5], int)))
    self.assertEqual(c.b, None)
    c.b = 3
    self.assertEqual(c.b, 3)
    with self.assertRaises(ValueError):
        c.b = 2
    self.assertEqual(c.b, 3)
    with self.assertRaises(ValueError):
        c.b = {}
    self.assertEqual(c.b, 3)
    c.b = '1'
    self.assertEqual(c.b, 1)

    class Container(object):

        def __init__(self, vals):
            self._vals = vals

        def __str__(self):
            return f'Container{self._vals}'

        def __contains__(self, val):
            return val in self._vals
    c.declare('c', ConfigValue(None, In(Container([1, 3, 5]))))
    self.assertEqual(c.get('c').domain_name(), 'In(Container[1, 3, 5])')
    self.assertEqual(c.c, None)
    c.c = 3
    self.assertEqual(c.c, 3)
    with self.assertRaises(ValueError):
        c.c = 2
    self.assertEqual(c.c, 3)