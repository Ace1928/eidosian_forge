import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def handle_semicolon(self, current_token):
    if self.start_of_statement(current_token):
        self._output.space_before_token = False
    else:
        self.handle_whitespace_and_comments(current_token)
    next_token = self._tokens.peek()
    while self._flags.mode == MODE.Statement and (not (self._flags.if_block and reserved_word(next_token, 'else'))) and (not self._flags.do_block):
        self.restore_mode()
    if self._flags.import_block:
        self._flags.import_block = False
    self.print_token(current_token)