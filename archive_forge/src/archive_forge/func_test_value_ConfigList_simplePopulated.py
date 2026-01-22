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
def test_value_ConfigList_simplePopulated(self):
    self.config['nodes'].append('1')
    self.config['nodes'].append(3)
    self.config['nodes'].append()
    val = self.config['nodes'].value()
    self.assertIs(type(val), list)
    self.assertEqual(len(val), 3)
    self.assertEqual(val, [1, 3, 0])