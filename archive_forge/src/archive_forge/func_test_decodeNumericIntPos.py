import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decodeNumericIntPos(self):
    input = '31337'
    self.assertEqual(31337, ujson.decode(input))