import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decimalDecodeTest(self):
    sut = {'a': 4.56}
    encoded = ujson.encode(sut)
    decoded = ujson.decode(encoded)
    self.assertAlmostEqual(sut[u'a'], decoded[u'a'])