import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeDictConversion(self):
    input = {'k1': 1, 'k2': 2, 'k3': 3, 'k4': 4}
    output = ujson.encode(input)
    self.assertEqual(input, json.loads(output))
    self.assertEqual(input, ujson.decode(output))
    self.assertEqual(input, ujson.decode(output))