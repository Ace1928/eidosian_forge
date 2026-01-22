import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decodeVeryTooBigValue(self):
    input = '18446744073709551616'
    self.assertRaises(ValueError, ujson.decode, input)