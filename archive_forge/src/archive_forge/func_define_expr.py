from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
def define_expr(s):
    from pyparsing import Literal, And, Word, alphas, nums, Optional, NotAny
    alpha_word = (~Literal('end') + Word(alphas, asKeyword=True)).setName('alpha')
    num_word = Word(nums, asKeyword=True).setName('int')
    ret = eval(s)
    ret.streamline()
    print_(ret)
    return ret