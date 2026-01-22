import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeNullCharacter(self):
    input = '31337 \x00 1337'
    output = ujson.encode(input)
    self.assertEqual(input, json.loads(output))
    self.assertEqual(output, json.dumps(input))
    self.assertEqual(input, ujson.decode(output))
    input = '\x00'
    output = ujson.encode(input)
    self.assertEqual(input, json.loads(output))
    self.assertEqual(output, json.dumps(input))
    self.assertEqual(input, ujson.decode(output))
    self.assertEqual('"  \\u0000\\r\\n "', ujson.dumps('  \x00\r\n '))