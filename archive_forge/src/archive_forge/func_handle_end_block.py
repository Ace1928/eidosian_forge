import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def handle_end_block(self, current_token):
    self.handle_whitespace_and_comments(current_token)
    while self._flags.mode == MODE.Statement:
        self.restore_mode()
    empty_braces = self._flags.last_token.type == TOKEN.START_BLOCK
    if self._flags.inline_frame and (not empty_braces):
        self._output.space_before_token = True
    elif self._options.brace_style == 'expand':
        if not empty_braces:
            self.print_newline()
    elif not empty_braces:
        if self.is_array(self._flags.mode) and self._options.keep_array_indentation:
            self._options.keep_array_indentation = False
            self.print_newline()
            self._options.keep_array_indentation = True
        else:
            self.print_newline()
    self.restore_mode()
    self.print_token(current_token)