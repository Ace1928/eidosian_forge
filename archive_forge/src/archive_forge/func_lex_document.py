from __future__ import unicode_literals
from prompt_toolkit.document import Document
from prompt_toolkit.layout.lexers import Lexer
from prompt_toolkit.layout.utils import split_lines
from prompt_toolkit.token import Token
from .compiler import _CompiledGrammar
from six.moves import range
def lex_document(self, cli, document):
    lines = list(split_lines(self._get_tokens(cli, document.text)))

    def get_line(lineno):
        try:
            return lines[lineno]
        except IndexError:
            return []
    return get_line