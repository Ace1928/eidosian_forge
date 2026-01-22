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
def test_block_items(self):
    ref = [('scenario file', 'Net3.tsg'), ('merlion', False), ('detection', [1, 2, 3])]
    items = self.config['scenario'].items()
    self.assertIsNot(type(items), list)
    self.assertEqual(list(items), ref)
    self.assertIsNot(items, self.config['scenario'].items())
    out = StringIO()
    with LoggingIntercept(out):
        itemiter = self.config['scenario'].iteritems()
        self.assertIsNot(itemiter, self.config['scenario'].iteritems())
    self.assertIn('The iteritems method is deprecated', out.getvalue())
    self.assertIsNot(type(itemiter), list)
    self.assertEqual(list(itemiter), ref)