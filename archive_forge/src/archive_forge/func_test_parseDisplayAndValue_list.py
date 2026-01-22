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
@unittest.skipIf(not yaml_available, 'Test requires PyYAML')
def test_parseDisplayAndValue_list(self):
    self.config['scenarios'].append()
    self.config['scenarios'].append({'merlion': True, 'detection': []})
    test = _display(self.config)
    sys.stdout.write(test)
    self.assertEqual(yaml_load(test), self.config.value())