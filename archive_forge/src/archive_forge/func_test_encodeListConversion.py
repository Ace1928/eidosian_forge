import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeListConversion(self):
    input = [1, 2, 3, 4]
    output = ujson.encode(input)
    self.assertEqual(input, json.loads(output))
    self.assertEqual(input, ujson.decode(output))