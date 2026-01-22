import unittest
from . import antlr_grammar
def testRule(self):
    text = 'expr    : term ( ( PLUS | MINUS )  term )* ;'
    antlr_grammar.rule.parseString(text)