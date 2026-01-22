import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeRecursionMax(self):

    class O2:
        member = 0

        def toDict(self):
            return {'member': self.member}

    class O1:
        member = 0

        def toDict(self):
            return {'member': self.member}
    input = O1()
    input.member = O2()
    input.member.member = input
    self.assertRaises(OverflowError, ujson.encode, input)