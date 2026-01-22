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
def getATNWithBypassAlts(self):
    serializedAtn = self.getSerializedATN()
    if serializedAtn is None:
        raise UnsupportedOperationException('The current parser does not support an ATN with bypass alternatives.')
    result = self.bypassAltsAtnCache.get(serializedAtn, None)
    if result is None:
        deserializationOptions = ATNDeserializationOptions()
        deserializationOptions.generateRuleBypassTransitions = True
        result = ATNDeserializer(deserializationOptions).deserialize(serializedAtn)
        self.bypassAltsAtnCache[serializedAtn] = result
    return result