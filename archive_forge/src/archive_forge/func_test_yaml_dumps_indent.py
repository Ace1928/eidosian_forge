from io import StringIO
from pathlib import Path
import pytest
from .._yaml_api import yaml_dumps, yaml_loads, read_yaml, write_yaml
from .._yaml_api import is_yaml_serializable
from ..ruamel_yaml.comments import CommentedMap
from .util import make_tempdir
def test_yaml_dumps_indent():
    data = {'a': [1, 'hello'], 'b': {'foo': 'bar', 'baz': [10.5, 120]}}
    result = yaml_dumps(data, indent_mapping=2, indent_sequence=2, indent_offset=0)
    expected = 'a:\n- 1\n- hello\nb:\n  foo: bar\n  baz:\n  - 10.5\n  - 120\n'
    assert result == expected