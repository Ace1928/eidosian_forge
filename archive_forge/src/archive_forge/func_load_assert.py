from __future__ import print_function, unicode_literals
import sys
import pytest  # NOQA
import warnings  # NOQA
from pathlib import Path
def load_assert(self, input, confirm, yaml_version=None):
    from srsly.ruamel_yaml.compat import Mapping
    d = self.yaml_load(input.value, yaml_version=yaml_version)[1]
    print('confirm.value', confirm.value, type(confirm.value))
    if isinstance(confirm.value, Mapping):
        r = range(confirm.value['range'])
        lines = confirm.value['lines'].splitlines()
        for idx in r:
            for line in lines:
                line = 'assert ' + line
                print(line)
                exec(line)
    else:
        for line in confirm.value.splitlines():
            line = 'assert ' + line
            print(line)
            exec(line)