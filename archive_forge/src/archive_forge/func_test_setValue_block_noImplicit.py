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
def test_setValue_block_noImplicit(self):
    _test = {'epanet file': 'no_file.inp', 'foo': 1}
    with self.assertRaisesRegex(ValueError, "key 'foo' not defined for ConfigDict 'network' and implicit"):
        self.config['network'] = _test
    self.assertEqual(self._reference, self.config.value())