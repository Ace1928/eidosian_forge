from __future__ import print_function
import sys
import textwrap
import pytest
from pathlib import Path
def test_dump_too_many_args(self, tmpdir):
    from srsly.ruamel_yaml import YAML
    fn = Path(str(tmpdir)) / 'test.yaml'
    yaml = YAML()
    data = yaml.map()
    data['a'] = 1
    data['b'] = 2
    with pytest.raises(TypeError):
        yaml.dump(data, fn, True)