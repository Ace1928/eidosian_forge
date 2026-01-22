import unittest
from . import antlr_grammar
def testTokensSpec(self):
    text = "tokens {\n                            PLUS     = '+' ;\n                            MINUS    = '-' ;\n                            MULT    = '*' ;\n                            DIV    = '/' ;\n                        }"
    antlr_grammar.tokensSpec.parseString(text)