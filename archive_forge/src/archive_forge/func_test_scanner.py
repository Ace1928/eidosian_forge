from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_scanner(self):

    def s_ident(scanner, token):
        return token

    def s_operator(scanner, token):
        return 'op%s' % token

    def s_float(scanner, token):
        return float(token)

    def s_int(scanner, token):
        return int(token)
    scanner = regex.Scanner([('[a-zA-Z_]\\w*', s_ident), ('\\d+\\.\\d*', s_float), ('\\d+', s_int), ('=|\\+|-|\\*|/', s_operator), ('\\s+', None)])
    self.assertEqual(repr(type(scanner.scanner.scanner('').pattern)), self.PATTERN_CLASS)
    self.assertEqual(scanner.scan('sum = 3*foo + 312.50 + bar'), (['sum', 'op=', 3, 'op*', 'foo', 'op+', 312.5, 'op+', 'bar'], ''))