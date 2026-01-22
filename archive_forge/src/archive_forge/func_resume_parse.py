from typing import Iterator, List
from copy import copy
import warnings
from lark.exceptions import UnexpectedToken
from lark.lexer import Token, LexerThread
def resume_parse(self):
    """Resume automated parsing from the current state.
        """
    return self.parser.parse_from_state(self.parser_state, last_token=self.lexer_thread.state.last_token)