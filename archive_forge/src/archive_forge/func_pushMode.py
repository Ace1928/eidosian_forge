from io import StringIO
import sys
from antlr4.CommonTokenFactory import CommonTokenFactory
from antlr4.atn.LexerATNSimulator import LexerATNSimulator
from antlr4.InputStream import InputStream
from antlr4.Recognizer import Recognizer
from antlr4.Token import Token
from antlr4.error.Errors import IllegalStateException, LexerNoViableAltException, RecognitionException
def pushMode(self, m: int):
    if self._interp.debug:
        print('pushMode ' + str(m), file=self._output)
    self._modeStack.append(self._mode)
    self.mode(m)