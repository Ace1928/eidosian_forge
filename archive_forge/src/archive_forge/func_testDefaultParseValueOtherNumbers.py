from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import parser
from fire import testutils
def testDefaultParseValueOtherNumbers(self):
    self.assertEqual(parser.DefaultParseValue('1e5'), 100000.0)