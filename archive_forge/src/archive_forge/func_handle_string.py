import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def handle_string(self, current_token):
    if current_token.text[0] == '`' and current_token.newlines == 0 and (current_token.whitespace_before == '') and (self._flags.last_token.type == TOKEN.WORD or current_token.previous.text == ')'):
        pass
    elif self.start_of_statement(current_token):
        self._output.space_before_token = True
    else:
        self.handle_whitespace_and_comments(current_token)
        if self._flags.last_token.type in [TOKEN.RESERVED, TOKEN.WORD] or self._flags.inline_frame:
            self._output.space_before_token = True
        elif self._flags.last_token.type in [TOKEN.COMMA, TOKEN.START_EXPR, TOKEN.EQUALS, TOKEN.OPERATOR]:
            if not self.start_of_object_property():
                self.allow_wrap_or_preserved_newline(current_token)
        elif current_token.text[0] == '`' and self._flags.last_token.type == TOKEN.END_EXPR and (current_token.previous.text in [']', ')']) and (current_token.newlines == 0):
            self._output.space_before_token = True
        else:
            self.print_newline()
    self.print_token(current_token)