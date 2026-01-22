import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_not(self) -> nodes.Expr:
    if self.stream.current.test('name:not'):
        lineno = next(self.stream).lineno
        return nodes.Not(self.parse_not(), lineno=lineno)
    return self.parse_compare()