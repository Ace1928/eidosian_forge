from __future__ import print_function
import sys
import pytest
def test_single_dump(self, capsys):
    from srsly.ruamel_yaml import YAML
    with YAML(output=sys.stdout) as yaml:
        yaml.dump(single_data)
    out, err = capsys.readouterr()
    print(err)
    assert out == single_doc