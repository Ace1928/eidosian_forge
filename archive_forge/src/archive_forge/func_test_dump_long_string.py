import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
@pytest.mark.parametrize('first_length', list(range(2, 7)))
@pytest.mark.parametrize('second_length', list(range(10919, 10924)))
def test_dump_long_string(first_length, second_length):
    ujson.dumps(['a' * first_length, '\x00' * second_length])