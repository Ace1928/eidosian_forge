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
def test_setItem_block_implicit_domain(self):
    ref = self._reference['scenario']
    ref['foo'] = '1'
    self.config['scenario']['foo'] = 1
    self.assertEqual(ref, self.config['scenario'].value())
    ref['bar'] = '1'
    self.config['scenario']['bar'] = 1
    self.assertEqual(ref, self.config['scenario'].value())