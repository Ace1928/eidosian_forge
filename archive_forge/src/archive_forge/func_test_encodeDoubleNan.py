import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeDoubleNan(self):
    input = float('nan')
    self.assertRaises(OverflowError, ujson.encode, input)