from __future__ import (absolute_import, division, print_function)
import importlib
import os
import re
import sys
import textwrap
import yaml
def recreate_lines(self):
    data = yaml.dump(self.data, default_flow_style=False, indent=4, Dumper=Dumper, sort_keys=False)
    self.lines = data.splitlines()