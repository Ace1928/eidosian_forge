import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeIntConversion(self):
    input = 31337
    output = ujson.encode(input)
    self.assertEqual(input, json.loads(output))
    self.assertEqual(output, json.dumps(input))
    self.assertEqual(input, ujson.decode(output))