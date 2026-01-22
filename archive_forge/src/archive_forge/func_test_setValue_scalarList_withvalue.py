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
def test_setValue_scalarList_withvalue(self):
    self.config['scenario']['detection'] = [6]
    val = self.config['scenario']['detection']
    self.assertIs(type(val), list)
    self.assertEqual(val, [6])