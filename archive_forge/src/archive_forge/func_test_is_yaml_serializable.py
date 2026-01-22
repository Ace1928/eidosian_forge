from io import StringIO
from pathlib import Path
import pytest
from .._yaml_api import yaml_dumps, yaml_loads, read_yaml, write_yaml
from .._yaml_api import is_yaml_serializable
from ..ruamel_yaml.comments import CommentedMap
from .util import make_tempdir
@pytest.mark.parametrize('obj,expected', [(['a', 'b', 1, 2], True), ({'a': 'b', 'c': 123}, True), ('hello', True), (lambda x: x, False), ({'a': lambda x: x}, False)])
def test_is_yaml_serializable(obj, expected):
    assert is_yaml_serializable(obj) == expected
    assert is_yaml_serializable(obj) == expected