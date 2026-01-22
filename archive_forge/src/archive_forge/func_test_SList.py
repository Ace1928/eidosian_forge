import os
import math
import random
from pathlib import Path
import pytest
from IPython.utils import text
def test_SList():
    sl = text.SList(['a 11', 'b 1', 'a 2'])
    assert sl.n == 'a 11\nb 1\na 2'
    assert sl.s == 'a 11 b 1 a 2'
    assert sl.grep(lambda x: x.startswith('a')) == text.SList(['a 11', 'a 2'])
    assert sl.fields(0) == text.SList(['a', 'b', 'a'])
    assert sl.sort(field=1, nums=True) == text.SList(['b 1', 'a 2', 'a 11'])