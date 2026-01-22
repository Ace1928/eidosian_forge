import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeDoubleInf(self):
    input = float('inf')
    self.assertRaises(OverflowError, ujson.encode, input)