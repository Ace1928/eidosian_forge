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
def test_read_json_file_invalid():
    file_contents = '{\n    "hello": world\n}'
    with make_tempdir({'tmp.json': file_contents}) as temp_dir:
        file_path = temp_dir / 'tmp.json'
        assert file_path.exists()
        with pytest.raises(ValueError):
            read_json(file_path)