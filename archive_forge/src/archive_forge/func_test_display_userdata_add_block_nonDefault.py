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
def test_display_userdata_add_block_nonDefault(self):
    self.config.add('foo', ConfigValue(0, int, None, None))
    self.config.add('bar', ConfigDict(implicit=True)).add('baz', ConfigDict())
    test = _display(self.config, 'userdata')
    sys.stdout.write(test)
    self.assertEqual(test, 'foo: 0\nbar:\n  baz:\n')