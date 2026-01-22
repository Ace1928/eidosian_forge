import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeUnicode4BytesUTF8Fail(self):
    input = b'\xfd\xbf\xbf\xbf\xbf\xbf'
    self.assertRaises(OverflowError, ujson.encode, input)