from __future__ import print_function
import sys
import pytest
def test_multi_dump(self, capsys):
    from srsly.ruamel_yaml import YAML
    with YAML(output=sys.stdout) as yaml:
        yaml.explicit_start = True
        yaml.dump(multi_doc_data[0])
        yaml.dump(multi_doc_data[1])
    out, err = capsys.readouterr()
    print(err)
    assert out == multi_doc