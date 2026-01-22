from io import StringIO
import sys
from antlr4.CommonTokenFactory import CommonTokenFactory
from antlr4.atn.LexerATNSimulator import LexerATNSimulator
from antlr4.InputStream import InputStream
from antlr4.Recognizer import Recognizer
from antlr4.Token import Token
from antlr4.error.Errors import IllegalStateException, LexerNoViableAltException, RecognitionException
def getErrorDisplay(self, s: str):
    with StringIO() as buf:
        for c in s:
            buf.write(self.getErrorDisplayForChar(c))
        return buf.getvalue()