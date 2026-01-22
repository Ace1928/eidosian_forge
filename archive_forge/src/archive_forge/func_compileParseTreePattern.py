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
def compileParseTreePattern(self, pattern: str, patternRuleIndex: int, lexer: Lexer=None):
    if lexer is None:
        if self.getTokenStream() is not None:
            tokenSource = self.getTokenStream().tokenSource
            if isinstance(tokenSource, Lexer):
                lexer = tokenSource
    if lexer is None:
        raise UnsupportedOperationException("Parser can't discover a lexer to use")
    m = ParseTreePatternMatcher(lexer, self)
    return m.compile(pattern, patternRuleIndex)