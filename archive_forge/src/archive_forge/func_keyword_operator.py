import collections
import re
import uuid
from ply import lex
from ply import yacc
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import lexer
from yaql.language import parser
from yaql.language import utils
@property
def keyword_operator(self):
    return self._keyword_operator