import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_dumpToFile(self):
    f = StringIO()
    ujson.dump([1, 2, 3], f)
    self.assertEqual('[1,2,3]', f.getvalue())