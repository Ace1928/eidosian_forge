from __future__ import absolute_import, division, print_function
import unittest
from datashader import datashape
from datashader.datashape import lexer
def check_failing_token(self, ds_str):
    self.assertRaises(datashape.DataShapeSyntaxError, list, lexer.lex(ds_str))