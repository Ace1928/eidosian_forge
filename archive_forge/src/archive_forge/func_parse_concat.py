import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_concat(self) -> nodes.Expr:
    lineno = self.stream.current.lineno
    args = [self.parse_math2()]
    while self.stream.current.type == 'tilde':
        next(self.stream)
        args.append(self.parse_math2())
    if len(args) == 1:
        return args[0]
    return nodes.Concat(args, lineno=lineno)