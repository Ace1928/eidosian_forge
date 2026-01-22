from io import StringIO
from typing import Callable
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import *
from antlr4.atn.Transition import *
from antlr4.atn.LexerAction import *
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
def readATN(self):
    idx = self.readInt()
    grammarType = ATNType.fromOrdinal(idx)
    maxTokenType = self.readInt()
    return ATN(grammarType, maxTokenType)