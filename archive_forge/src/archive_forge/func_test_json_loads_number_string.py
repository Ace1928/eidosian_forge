import pytest
from io import StringIO
from pathlib import Path
import gzip
import numpy
from .._json_api import (
from .._json_api import write_gzip_json, json_dumps, is_json_serializable
from .._json_api import json_loads
from ..util import force_string
from .util import make_tempdir
@pytest.mark.parametrize('obj,expected', [('-32', -32), ('32', 32), ('0', 0), ('-0', 0)])
def test_json_loads_number_string(obj, expected):
    assert json_loads(obj) == expected