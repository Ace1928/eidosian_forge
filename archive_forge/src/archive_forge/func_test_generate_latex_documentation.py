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
def test_generate_latex_documentation(self):
    cfg = ConfigDict()
    cfg.declare('int', ConfigValue(domain=int, default=10, doc='This is an integer parameter'))
    cfg.declare('in', ConfigValue(domain=In([1, 3, 5]), default=1, description='This parameter must be in {1,3,5}'))
    cfg.declare('lambda', ConfigValue(domain=lambda x: int(x), default=1, description='This is a float', doc='This parameter is actually a float, but for testing purposes we will use a lambda function for validation'))
    cfg.declare('list', ConfigList(domain=str, description='A simple list of strings'))
    self.assertEqual(cfg.generate_documentation(format='latex').strip(), '\n\\begin{description}[topsep=0pt,parsep=0.5em,itemsep=-0.4em]\n  \\item[{int}]\\hfill\n    \\\\This is an integer parameter\n  \\item[{in}]\\hfill\n    \\\\This parameter must be in {1,3,5}\n  \\item[{lambda}]\\hfill\n    \\\\This parameter is actually a float, but for testing purposes we will use\n    a lambda function for validation\n  \\item[{list}]\\hfill\n    \\\\A simple list of strings\n\\end{description}\n            '.strip())