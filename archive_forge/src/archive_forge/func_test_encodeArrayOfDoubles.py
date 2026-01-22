import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeArrayOfDoubles(self):
    input = [31337.31337, 31337.31337, 31337.31337, 31337.31337] * 10
    output = ujson.encode(input)
    self.assertEqual(input, json.loads(output))
    self.assertEqual(input, ujson.decode(output))