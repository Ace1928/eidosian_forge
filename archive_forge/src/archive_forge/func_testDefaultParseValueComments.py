from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import parser
from fire import testutils
def testDefaultParseValueComments(self):
    self.assertEqual(parser.DefaultParseValue('"0#comments"'), '0#comments')
    self.assertEqual(parser.DefaultParseValue('0#comments'), 0)