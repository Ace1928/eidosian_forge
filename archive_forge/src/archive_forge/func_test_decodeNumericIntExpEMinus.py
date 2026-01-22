import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decodeNumericIntExpEMinus(self):
    input = '1.337E-4'
    output = ujson.decode(input)
    self.assertEqual(output, json.loads(input))