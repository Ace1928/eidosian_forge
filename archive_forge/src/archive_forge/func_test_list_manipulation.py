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
def test_list_manipulation(self):
    self.assertEqual(len(self.config['scenarios']), 0)
    self.config['scenarios'].append()
    os = StringIO()
    with LoggingIntercept(os):
        self.config['scenarios'].add()
    self.assertIn('ConfigList.add() has been deprecated.  Use append()', os.getvalue())
    self.assertEqual(len(self.config['scenarios']), 2)
    self.config['scenarios'].append({'merlion': True, 'detection': []})
    self.assertEqual(len(self.config['scenarios']), 3)
    test = _display(self.config, 'userdata')
    sys.stdout.write(test)
    self.assertEqual(test, 'scenarios:\n  -\n  -\n  -\n    merlion: true\n    detection: []\n')
    self.config['scenarios'][0] = {'merlion': True, 'detection': []}
    self.assertEqual(len(self.config['scenarios']), 3)
    test = _display(self.config, 'userdata')
    sys.stdout.write(test)
    self.assertEqual(test, 'scenarios:\n  -\n    merlion: true\n    detection: []\n  -\n  -\n    merlion: true\n    detection: []\n')
    test = _display(self.config['scenarios'])
    sys.stdout.write(test)
    self.assertEqual(test, '-\n  scenario file: Net3.tsg\n  merlion: true\n  detection: []\n-\n  scenario file: Net3.tsg\n  merlion: false\n  detection: [1, 2, 3]\n-\n  scenario file: Net3.tsg\n  merlion: true\n  detection: []\n')