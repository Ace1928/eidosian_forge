from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import optparse
import sys
import antlr3
from six.moves import input
def parseOptions(self, argv):
    optParser = optparse.OptionParser()
    optParser.add_option('--encoding', action='store', type='string', dest='encoding')
    optParser.add_option('--input', action='store', type='string', dest='input')
    optParser.add_option('--interactive', '-i', action='store_true', dest='interactive')
    optParser.add_option('--no-output', action='store_true', dest='no_output')
    optParser.add_option('--profile', action='store_true', dest='profile')
    optParser.add_option('--hotshot', action='store_true', dest='hotshot')
    self.setupOptions(optParser)
    return optParser.parse_args(argv[1:])