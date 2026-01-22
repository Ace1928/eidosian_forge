import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeWithDecimal(self):
    input = 1.0
    output = ujson.encode(input)
    self.assertEqual(output, '1.0')