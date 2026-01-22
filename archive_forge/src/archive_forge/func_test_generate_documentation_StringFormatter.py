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
def test_generate_documentation_StringFormatter(self):
    CONFIG = ExampleConfig()
    doc = CONFIG.generate_documentation(format=String_ConfigFormatter(block_start='', block_end='', item_start='%s\n', item_body='%s', item_end='\n'), indent_spacing=4, width=66)
    ref = '    option_1\n        The first configuration option\n\n    solver_options\n\n        solver_option_1\n            The first solver configuration option\n\n        solver_option_2\n            The second solver configuration option\n\n        With a very long line containing\n        wrappable text in a long, silly paragraph\n        with little actual information.\n        #) but a bulleted list\n        #) with two bullets\n\n        solver_option_3\n            The third solver configuration option\n\n            This has a leading newline and a very long line containing\n            wrappable text in a long, silly paragraph with\n            little actual information.\n\n         .. and_a_list::\n            #) but a bulleted list\n            #) with two bullets\n\n    option_2\n        The second solver configuration option with a very long\n        line containing wrappable text in a long, silly paragraph\n        with little actual information.\n\n'
    self.assertEqual([_.rstrip() for _ in ref.splitlines()], [_.rstrip() for _ in doc.splitlines()])