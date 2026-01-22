from __future__ import print_function
import sys
import textwrap
import pytest
from pathlib import Path
def test_issue_135(self):
    from srsly.ruamel_yaml import YAML
    data = {'a': 1, 'b': 2}
    yaml = YAML(typ='safe')
    yaml.dump(data, sys.stdout)