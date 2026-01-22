import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_call_args(self) -> t.Tuple:
    token = self.stream.expect('lparen')
    args = []
    kwargs = []
    dyn_args = None
    dyn_kwargs = None
    require_comma = False

    def ensure(expr: bool) -> None:
        if not expr:
            self.fail('invalid syntax for function call expression', token.lineno)
    while self.stream.current.type != 'rparen':
        if require_comma:
            self.stream.expect('comma')
            if self.stream.current.type == 'rparen':
                break
        if self.stream.current.type == 'mul':
            ensure(dyn_args is None and dyn_kwargs is None)
            next(self.stream)
            dyn_args = self.parse_expression()
        elif self.stream.current.type == 'pow':
            ensure(dyn_kwargs is None)
            next(self.stream)
            dyn_kwargs = self.parse_expression()
        elif self.stream.current.type == 'name' and self.stream.look().type == 'assign':
            ensure(dyn_kwargs is None)
            key = self.stream.current.value
            self.stream.skip(2)
            value = self.parse_expression()
            kwargs.append(nodes.Keyword(key, value, lineno=value.lineno))
        else:
            ensure(dyn_args is None and dyn_kwargs is None and (not kwargs))
            args.append(self.parse_expression())
        require_comma = True
    self.stream.expect('rparen')
    return (args, kwargs, dyn_args, dyn_kwargs)