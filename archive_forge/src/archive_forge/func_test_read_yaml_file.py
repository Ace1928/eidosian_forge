from io import StringIO
from pathlib import Path
import pytest
from .._yaml_api import yaml_dumps, yaml_loads, read_yaml, write_yaml
from .._yaml_api import is_yaml_serializable
from ..ruamel_yaml.comments import CommentedMap
from .util import make_tempdir
def test_read_yaml_file():
    file_contents = 'a:\n- 1\n- hello\nb:\n  foo: bar\n  baz:\n  - 10.5\n  - 120\n'
    with make_tempdir({'tmp.yaml': file_contents}) as temp_dir:
        file_path = temp_dir / 'tmp.yaml'
        assert file_path.exists()
        data = read_yaml(file_path)
    assert len(data) == 2
    assert data['a'] == [1, 'hello']