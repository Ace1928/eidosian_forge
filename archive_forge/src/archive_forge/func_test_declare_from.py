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
def test_declare_from(self):
    cfg = ConfigDict()
    cfg.declare('a', ConfigValue(default=1, domain=int))
    cfg.declare('b', ConfigValue(default=2, domain=int))
    cfg2 = ConfigDict()
    cfg2.declare_from(cfg)
    self.assertEqual(cfg.value(), cfg2.value())
    self.assertIsNot(cfg.get('a'), cfg2.get('a'))
    self.assertIsNot(cfg.get('b'), cfg2.get('b'))
    cfg2 = ConfigDict()
    cfg2.declare_from(cfg, skip={'a'})
    self.assertEqual(cfg.value()['b'], cfg2.value()['b'])
    self.assertNotIn('a', cfg2)
    with self.assertRaisesRegex(ValueError, "passed a block with a duplicate field, 'b'"):
        cfg2.declare_from(cfg)
    with self.assertRaisesRegex(ValueError, 'only accepts other ConfigDicts'):
        cfg2.declare_from({})