import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_doubleLongDecimalIssue(self):
    sut = {'a': -12345678901234.568}
    encoded = json.dumps(sut)
    decoded = json.loads(encoded)
    self.assertEqual(sut, decoded)
    encoded = ujson.encode(sut)
    decoded = ujson.decode(encoded)
    self.assertEqual(sut, decoded)