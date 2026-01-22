import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_loadFileLikeObject(self):

    class filelike:

        def read(self):
            try:
                self.end
            except AttributeError:
                self.end = True
                return '[1,2,3,4]'
    f = filelike()
    self.assertEqual([1, 2, 3, 4], ujson.load(f))