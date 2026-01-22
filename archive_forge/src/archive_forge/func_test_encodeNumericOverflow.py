import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeNumericOverflow(self):
    self.assertRaises(OverflowError, ujson.encode, 12839128391289382193812939)