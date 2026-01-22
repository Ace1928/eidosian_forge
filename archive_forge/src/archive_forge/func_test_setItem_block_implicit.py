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
def test_setItem_block_implicit(self):
    ref = self._reference
    ref['foo'] = 1
    self.config['foo'] = 1
    self.assertEqual(ref, self.config.value())
    ref['bar'] = 1
    self.config['bar'] = 1
    self.assertEqual(ref, self.config.value())