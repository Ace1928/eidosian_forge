import unittest
from . import antlr_grammar
def testLexerRule2(self):
    text = "WHITESPACE : ( '\t' | ' ' | '\r' | '\n'| '\x0c' )+     { $channel = HIDDEN; } ;"