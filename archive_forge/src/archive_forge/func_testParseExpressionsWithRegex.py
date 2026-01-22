from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
def testParseExpressionsWithRegex(self):
    from itertools import product
    match_empty_regex = pp.Regex('[a-z]*')
    match_nonempty_regex = pp.Regex('[a-z]+')
    parser_classes = pp.ParseExpression.__subclasses__()
    test_string = 'abc def'
    expected = ['abc']
    for expr, cls in product((match_nonempty_regex, match_empty_regex), parser_classes):
        print_(expr, cls)
        parser = cls([expr])
        parsed_result = parser.parseString(test_string)
        print_(parsed_result.dump())
        self.assertParseResultsEquals(parsed_result, expected)
    for expr, cls in product((match_nonempty_regex, match_empty_regex), (pp.MatchFirst, pp.Or)):
        parser = cls([expr, expr])
        print_(parser)
        parsed_result = parser.parseString(test_string)
        print_(parsed_result.dump())
        self.assertParseResultsEquals(parsed_result, expected)