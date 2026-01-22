from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import parser
from fire import testutils
def testDefaultParseValueBool(self):
    self.assertEqual(parser.DefaultParseValue('True'), True)
    self.assertEqual(parser.DefaultParseValue('False'), False)