import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def handle_comma(self, current_token):
    self.handle_whitespace_and_comments(current_token, True)
    self.print_token(current_token)
    self._output.space_before_token = True
    if self._flags.declaration_statement:
        if self.is_expression(self._flags.parent.mode):
            self._flags.declaration_assignment = False
        if self._flags.declaration_assignment:
            self._flags.declaration_assignment = False
            self.print_newline(preserve_statement_flags=True)
        elif self._options.comma_first:
            self.allow_wrap_or_preserved_newline(current_token)
    elif self._flags.mode == MODE.ObjectLiteral or (self._flags.mode == MODE.Statement and self._flags.parent.mode == MODE.ObjectLiteral):
        if self._flags.mode == MODE.Statement:
            self.restore_mode()
        if not self._flags.inline_frame:
            self.print_newline()
    elif self._options.comma_first:
        self.allow_wrap_or_preserved_newline(current_token)