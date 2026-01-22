from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_chunksize_is_incremental():
    jsonl = '{"a": 1, "b": 2}\n        {"a": 3, "b": 4}\n        {"a": 5, "b": 6}\n        {"a": 7, "b": 8}\n' * 1000

    class MyReader:

        def __init__(self, contents) -> None:
            self.read_count = 0
            self.stringio = StringIO(contents)

        def read(self, *args):
            self.read_count += 1
            return self.stringio.read(*args)

        def __iter__(self) -> Iterator:
            self.read_count += 1
            return iter(self.stringio)
    reader = MyReader(jsonl)
    assert len(list(read_json(reader, lines=True, chunksize=100))) > 1
    assert reader.read_count > 10