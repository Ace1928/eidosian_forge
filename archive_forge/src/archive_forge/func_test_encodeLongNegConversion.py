import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeLongNegConversion(self):
    input = -9223372036854775808
    output = ujson.encode(input)
    json.loads(output)
    ujson.decode(output)
    self.assertEqual(input, json.loads(output))
    self.assertEqual(output, json.dumps(input))
    self.assertEqual(input, ujson.decode(output))