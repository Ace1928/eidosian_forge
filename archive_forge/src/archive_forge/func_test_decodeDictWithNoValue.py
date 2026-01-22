import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decodeDictWithNoValue(self):
    input = '{{{{"key":}}}}'
    self.assertRaises(ValueError, ujson.decode, input)