import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def handle_dot(self, current_token):
    if self.start_of_statement(current_token):
        pass
    else:
        self.handle_whitespace_and_comments(current_token, True)
    if re.search('^([0-9])+$', self._flags.last_token.text):
        self._output.space_before_token = True
    if reserved_array(self._flags.last_token, _special_word_set):
        self._output.space_before_token = False
    else:
        self.allow_wrap_or_preserved_newline(current_token, self._flags.last_token.text == ')' and self._options.break_chained_methods)
    if self._options.unindent_chained_methods and self._output.just_added_newline():
        self.deindent()
    self.print_token(current_token)