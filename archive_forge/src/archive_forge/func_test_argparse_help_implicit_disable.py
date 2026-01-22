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
def test_argparse_help_implicit_disable(self):
    self.config['scenario'].declare('epanet', ConfigValue(True, bool, 'Use EPANET as the Water quality model', None)).declare_as_argument(group='Scenario definition')
    parser = argparse.ArgumentParser(prog='tester')
    self.config.initialize_argparse(parser)
    help = parser.format_help()
    self.maxDiff = None
    self.assertIn("\n  -h, --help            show this help message and exit\n  --epanet-file EPANET  EPANET network inp file\n\nScenario definition:\n  --scenario-file STR   Scenario generation file, see the TEVASIM\n                        documentation\n  --merlion             Water quality model\n  --disable-epanet      [DON'T] Use EPANET as the Water quality model\n", help)