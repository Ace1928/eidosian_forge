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
def test_display_default(self):
    reference = 'network:\n  epanet file: Net3.inp\nscenario:\n  scenario file: Net3.tsg\n  merlion: false\n  detection: [1, 2, 3]\nscenarios: []\nnodes: []\nimpact:\n  metric: MC\nflushing:\n  flush nodes:\n    feasible nodes: ALL\n    infeasible nodes: NONE\n    max nodes: 2\n    rate: 600.0\n    response time: 60.0\n    duration: 600.0\n  close valves:\n    feasible pipes: ALL\n    infeasible pipes: NONE\n    max pipes: 2\n    response time: 60.0\n'
    test = _display(self.config)
    sys.stdout.write(test)
    self.assertEqual(test, reference)