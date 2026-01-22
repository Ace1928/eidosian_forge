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
def test_setValue_block_simplevalue(self):
    _test = {'merlion': True, 'detection': [1]}
    ref = self._reference['scenario']
    ref.update(_test)
    self.config['scenario'] = _test
    self.assertEqual(ref, self.config['scenario'].value())