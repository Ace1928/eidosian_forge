import sys
from antlr4.BufferedTokenStream import TokenStream
from antlr4.CommonTokenFactory import TokenFactory
from antlr4.error.ErrorStrategy import DefaultErrorStrategy
from antlr4.InputStream import InputStream
from antlr4.Recognizer import Recognizer
from antlr4.RuleContext import RuleContext
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Token import Token
from antlr4.Lexer import Lexer
from antlr4.atn.ATNDeserializer import ATNDeserializer
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
from antlr4.error.Errors import UnsupportedOperationException, RecognitionException
from antlr4.tree.ParseTreePatternMatcher import ParseTreePatternMatcher
from antlr4.tree.Tree import ParseTreeListener, TerminalNode, ErrorNode
def dumpDFA(self):
    seenOne = False
    for i in range(0, len(self._interp.decisionToDFA)):
        dfa = self._interp.decisionToDFA[i]
        if len(dfa.states) > 0:
            if seenOne:
                print(file=self._output)
            print('Decision ' + str(dfa.decision) + ':', file=self._output)
            print(dfa.toString(self.literalNames, self.symbolicNames), end='', file=self._output)
            seenOne = True