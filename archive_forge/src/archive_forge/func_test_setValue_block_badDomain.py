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
def test_setValue_block_badDomain(self):
    _test = {'merlion': True, 'detection': ['a'], 'foo': 1, 'a': 1}
    with self.assertRaisesRegex(ValueError, 'invalid value for configuration'):
        self.config['scenario'] = _test
    self.assertEqual(self._reference, self.config.value())
    with self.assertRaisesRegex(ValueError, 'Expected dict value for scenario.set_value, found list'):
        self.config['scenario'] = []
    self.assertEqual(self._reference, self.config.value())