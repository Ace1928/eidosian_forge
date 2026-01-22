from copy import deepcopy, copy
from typing import Dict, Any, Generic, List
from ..lexer import Token, LexerThread
from ..common import ParserCallbacks
from .lalr_analysis import Shift, ParseTableBase, StateT
from lark.exceptions import UnexpectedToken
@property
def position(self) -> StateT:
    return self.state_stack[-1]