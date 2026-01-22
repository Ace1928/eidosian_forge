import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def print_token(self, current_token, s=None):
    if self._output.raw:
        self._output.add_raw_token(current_token)
        return
    if self._options.comma_first and current_token.previous and (current_token.previous.type == TOKEN.COMMA) and self._output.just_added_newline():
        if self._output.previous_line.last() == ',':
            popped = self._output.previous_line.pop()
            if self._output.previous_line.is_empty():
                self._output.previous_line.push(popped)
                self._output.trim(True)
                self._output.current_line.pop()
                self._output.trim()
            self.print_token_line_indentation(current_token)
            self._output.add_token(',')
            self._output.space_before_token = True
    if s is None:
        s = current_token.text
    self.print_token_line_indentation(current_token)
    self._output.non_breaking_space = True
    self._output.add_token(s)
    if self._output.previous_token_wrapped:
        self._flags.multiline_frame = True