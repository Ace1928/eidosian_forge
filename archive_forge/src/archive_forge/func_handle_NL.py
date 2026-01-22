from abc import ABC, abstractmethod
from typing import List, Iterator
from .exceptions import LarkError
from .lark import PostLex
from .lexer import Token
def handle_NL(self, token: Token) -> Iterator[Token]:
    if self.paren_level > 0:
        return
    yield token
    indent_str = token.rsplit('\n', 1)[1]
    indent = indent_str.count(' ') + indent_str.count('\t') * self.tab_len
    if indent > self.indent_level[-1]:
        self.indent_level.append(indent)
        yield Token.new_borrow_pos(self.INDENT_type, indent_str, token)
    else:
        while indent < self.indent_level[-1]:
            self.indent_level.pop()
            yield Token.new_borrow_pos(self.DEDENT_type, indent_str, token)
        if indent != self.indent_level[-1]:
            raise DedentError('Unexpected dedent to column %s. Expected dedent to %s' % (indent, self.indent_level[-1]))