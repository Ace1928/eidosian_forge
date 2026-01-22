import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decodeNumericIntExpe(self):
    input = '1337e40'
    output = ujson.decode(input)
    self.assertEqual(output, json.loads(input))