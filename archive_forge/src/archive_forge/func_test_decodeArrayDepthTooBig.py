import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decodeArrayDepthTooBig(self):
    input = '[' * (1024 * 1024)
    self.assertRaises(ValueError, ujson.decode, input)