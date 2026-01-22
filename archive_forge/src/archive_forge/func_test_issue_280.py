from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_280(self):
    from srsly.ruamel_yaml import YAML
    from srsly.ruamel_yaml.representer import RepresenterError
    from collections import namedtuple
    from sys import stdout
    T = namedtuple('T', ('a', 'b'))
    t = T(1, 2)
    yaml = YAML()
    with pytest.raises(RepresenterError, match='cannot represent'):
        yaml.dump({'t': t}, stdout)