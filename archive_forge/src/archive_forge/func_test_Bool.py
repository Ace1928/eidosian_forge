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
def test_Bool(self):
    c = ConfigDict()
    c.declare('a', ConfigValue(True, Bool))
    self.assertEqual(c.a, True)
    c.a = False
    self.assertEqual(c.a, False)
    c.a = 1
    self.assertEqual(c.a, True)
    c.a = 'n'
    self.assertEqual(c.a, False)
    c.a = 'T'
    self.assertEqual(c.a, True)
    c.a = 'no'
    self.assertEqual(c.a, False)
    c.a = '1'
    self.assertEqual(c.a, True)
    c.a = 0.0
    self.assertEqual(c.a, False)
    c.a = True
    self.assertEqual(c.a, True)
    c.a = 0
    self.assertEqual(c.a, False)
    c.a = 'y'
    self.assertEqual(c.a, True)
    c.a = 'F'
    self.assertEqual(c.a, False)
    c.a = 'yes'
    self.assertEqual(c.a, True)
    c.a = '0'
    self.assertEqual(c.a, False)
    c.a = 1.0
    self.assertEqual(c.a, True)
    with self.assertRaises(ValueError):
        c.a = 2
    self.assertEqual(c.a, True)
    with self.assertRaises(ValueError):
        c.a = 'a'
    self.assertEqual(c.a, True)
    with self.assertRaises(ValueError):
        c.a = 0.5
    self.assertEqual(c.a, True)