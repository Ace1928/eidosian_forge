import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decodeWithTrailingNonWhitespaces(self):
    input = '{}\n\t a'
    self.assertRaises(ValueError, ujson.decode, input)