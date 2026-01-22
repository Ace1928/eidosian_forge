import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_math2(self) -> nodes.Expr:
    lineno = self.stream.current.lineno
    left = self.parse_pow()
    while self.stream.current.type in ('mul', 'div', 'floordiv', 'mod'):
        cls = _math_nodes[self.stream.current.type]
        next(self.stream)
        right = self.parse_pow()
        left = cls(left, right, lineno=lineno)
        lineno = self.stream.current.lineno
    return left