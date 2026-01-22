import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def handle_token(self, current_token, preserve_statement_flags=False):
    if current_token.type == TOKEN.START_EXPR:
        self.handle_start_expr(current_token)
    elif current_token.type == TOKEN.END_EXPR:
        self.handle_end_expr(current_token)
    elif current_token.type == TOKEN.START_BLOCK:
        self.handle_start_block(current_token)
    elif current_token.type == TOKEN.END_BLOCK:
        self.handle_end_block(current_token)
    elif current_token.type == TOKEN.WORD:
        self.handle_word(current_token)
    elif current_token.type == TOKEN.RESERVED:
        self.handle_word(current_token)
    elif current_token.type == TOKEN.SEMICOLON:
        self.handle_semicolon(current_token)
    elif current_token.type == TOKEN.STRING:
        self.handle_string(current_token)
    elif current_token.type == TOKEN.EQUALS:
        self.handle_equals(current_token)
    elif current_token.type == TOKEN.OPERATOR:
        self.handle_operator(current_token)
    elif current_token.type == TOKEN.COMMA:
        self.handle_comma(current_token)
    elif current_token.type == TOKEN.BLOCK_COMMENT:
        self.handle_block_comment(current_token, preserve_statement_flags)
    elif current_token.type == TOKEN.COMMENT:
        self.handle_comment(current_token, preserve_statement_flags)
    elif current_token.type == TOKEN.DOT:
        self.handle_dot(current_token)
    elif current_token.type == TOKEN.EOF:
        self.handle_eof(current_token)
    elif current_token.type == TOKEN.UNKNOWN:
        self.handle_unknown(current_token, preserve_statement_flags)
    else:
        self.handle_unknown(current_token, preserve_statement_flags)