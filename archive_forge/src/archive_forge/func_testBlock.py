import unittest
from . import antlr_grammar
def testBlock(self):
    text = '( PLUS | MINUS )'
    antlr_grammar.block.parseString(text)