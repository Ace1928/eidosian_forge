import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeDoubleConversion(self):
    input = math.pi
    output = ujson.encode(input)
    self.assertEqual(round(input, 5), round(json.loads(output), 5))
    self.assertEqual(round(input, 5), round(ujson.decode(output), 5))