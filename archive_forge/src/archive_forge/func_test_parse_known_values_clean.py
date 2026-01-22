import sys
from parser import parse, MalformedQueryStringError
from builder import build
import unittest
def test_parse_known_values_clean(self):
    """parse should give known result with known input"""
    self.maxDiff = None
    for dic in self.knownValuesClean:
        result = parse(build(dic), unquote=True)
        self.assertEqual(dic, result)