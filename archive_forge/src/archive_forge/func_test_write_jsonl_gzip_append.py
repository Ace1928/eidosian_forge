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
def test_write_jsonl_gzip_append():
    """Tests appending data to a gzipped .jsonl file."""
    data = [{'hello': 'world'}, {'test': 123}]
    expected = ['{"hello":"world"}\n', '{"test":123}\n', '\n', '{"hello":"world"}\n', '{"test":123}\n']
    with make_tempdir() as temp_dir:
        file_path = temp_dir / 'tmp.json'
        write_gzip_jsonl(file_path, data)
        write_gzip_jsonl(file_path, data, append=True)
        with gzip.open(file_path, 'r') as f:
            assert [line.decode('utf8') for line in f.readlines()] == expected