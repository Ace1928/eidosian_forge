import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def testEncodeSymbols(self):
    s = '✿♡✿'
    encoded = ujson.dumps(s)
    encoded_json = json.dumps(s)
    self.assertEqual(len(encoded), len(s) * 6 + 2)
    self.assertEqual(encoded, encoded_json)
    decoded = ujson.loads(encoded)
    self.assertEqual(s, decoded)
    encoded = ujson.dumps(s, ensure_ascii=False)
    encoded_json = json.dumps(s, ensure_ascii=False)
    self.assertEqual(len(encoded), len(s) + 2)
    self.assertEqual(encoded, encoded_json)
    decoded = ujson.loads(encoded)
    self.assertEqual(s, decoded)