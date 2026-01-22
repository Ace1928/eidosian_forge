from io import StringIO
import sys
from antlr4.CommonTokenFactory import CommonTokenFactory
from antlr4.atn.LexerATNSimulator import LexerATNSimulator
from antlr4.InputStream import InputStream
from antlr4.Recognizer import Recognizer
from antlr4.Token import Token
from antlr4.error.Errors import IllegalStateException, LexerNoViableAltException, RecognitionException
def emitEOF(self):
    cpos = self.column
    lpos = self.line
    eof = self._factory.create(self._tokenFactorySourcePair, Token.EOF, None, Token.DEFAULT_CHANNEL, self._input.index, self._input.index - 1, lpos, cpos)
    self.emitToken(eof)
    return eof