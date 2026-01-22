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
def test_getItem_setItem(self):
    self.assertFalse(self.config._userAccessed)
    self.assertFalse(self.config._data['scenario']._userAccessed)
    self.assertFalse(self.config._data['scenario']._data['detection']._userAccessed)
    self.assertFalse(self.config['scenario'].get('detection')._userAccessed)
    self.assertTrue(self.config._userAccessed)
    self.assertTrue(self.config._data['scenario']._userAccessed)
    self.assertFalse(self.config._data['scenario']._data['detection']._userAccessed)
    self.assertFalse(self.config._userSet)
    self.assertFalse(self.config._data['scenario']._userSet)
    self.assertFalse(self.config['scenario']._data['detection']._userSet)
    self.assertEqual(self.config['scenario']['detection'], [1, 2, 3])
    self.config['scenario']['detection'] = [42.5]
    self.assertEqual(self.config['scenario']['detection'], [42])
    self.assertFalse(self.config._userSet)
    self.assertFalse(self.config._data['scenario']._userSet)
    self.assertTrue(self.config['scenario'].get('detection')._userSet)