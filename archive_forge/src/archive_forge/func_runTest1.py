from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
def runTest1(self):
    from pyparsing import Optional, Keyword
    for the_input in ['Tal Weiss Major', 'Tal Major', 'Weiss Major', 'Major', 'Major Tal', 'Major Weiss', 'Major Tal Weiss']:
        print_(the_input)
        parser1 = Optional('Tal') + Optional('Weiss') & Keyword('Major')
        parser2 = Optional(Optional('Tal') + Optional('Weiss')) & Keyword('Major')
        p1res = parser1.parseString(the_input)
        p2res = parser2.parseString(the_input)
        self.assertEqual(p1res.asList(), p2res.asList(), 'Each failed to match with nested Optionals, ' + str(p1res.asList()) + ' should match ' + str(p2res.asList()))