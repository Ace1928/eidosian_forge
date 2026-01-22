from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import optparse
import sys
import antlr3
from six.moves import input
def setupOptions(self, optParser):
    optParser.add_option('--lexer', action='store', type='string', dest='lexerClass', default=None)
    optParser.add_option('--parser', action='store', type='string', dest='parserClass', default=None)
    optParser.add_option('--parser-rule', action='store', type='string', dest='parserRule', default=None)
    optParser.add_option('--rule', action='store', type='string', dest='walkerRule')