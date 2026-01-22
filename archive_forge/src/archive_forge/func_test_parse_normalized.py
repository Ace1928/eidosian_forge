import sys
from parser import parse, MalformedQueryStringError
from builder import build
import unittest
def test_parse_normalized(self):
    result = parse(build(self.knownValues), normalized=True)
    self.assertEqual(self.knownValuesNormalized, result)