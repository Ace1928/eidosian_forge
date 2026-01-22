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
def test_self_assignment(self):
    cfg = ConfigDict()
    self.assertNotIn('d', dir(cfg))
    cfg.d = cfg.declare('d', ConfigValue(10, int))
    self.assertIn('d', dir(cfg))
    cfg.aa = cfg.declare('aa', ConfigValue(1, int))
    self.assertIn('aa', dir(cfg))
    self.assertEqual(dir(cfg), sorted(dir(cfg)))
    with self.assertRaisesRegex(ValueError, "Key 'b' not defined in ConfigDict ''"):
        cfg.b = cfg.declare('bb', ConfigValue(2, int))