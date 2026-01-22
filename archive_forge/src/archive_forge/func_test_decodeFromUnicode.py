import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decodeFromUnicode(self):
    input = '{"obj": 31337}'
    dec1 = ujson.decode(input)
    dec2 = ujson.decode(str(input))
    self.assertEqual(dec1, dec2)