from io import StringIO
from typing import Callable
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import *
from antlr4.atn.Transition import *
from antlr4.atn.LexerAction import *
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
def stateFactory(self, type: int, ruleIndex: int):
    if type > len(self.stateFactories) or self.stateFactories[type] is None:
        raise Exception('The specified state type ' + str(type) + ' is not valid.')
    else:
        s = self.stateFactories[type]()
        if s is not None:
            s.ruleIndex = ruleIndex
    return s