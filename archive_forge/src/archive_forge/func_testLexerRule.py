import unittest
from . import antlr_grammar
def testLexerRule(self):
    text = "fragment DIGIT    : '0'..'9' ;"
    antlr_grammar.rule.parseString(text)