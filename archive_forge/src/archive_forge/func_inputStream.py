from io import StringIO
import sys
from antlr4.CommonTokenFactory import CommonTokenFactory
from antlr4.atn.LexerATNSimulator import LexerATNSimulator
from antlr4.InputStream import InputStream
from antlr4.Recognizer import Recognizer
from antlr4.Token import Token
from antlr4.error.Errors import IllegalStateException, LexerNoViableAltException, RecognitionException
@inputStream.setter
def inputStream(self, input: InputStream):
    self._input = None
    self._tokenFactorySourcePair = (self, self._input)
    self.reset()
    self._input = input
    self._tokenFactorySourcePair = (self, self._input)