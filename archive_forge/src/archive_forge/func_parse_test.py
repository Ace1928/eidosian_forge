import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_test(self, node: nodes.Expr) -> nodes.Expr:
    token = next(self.stream)
    if self.stream.current.test('name:not'):
        next(self.stream)
        negated = True
    else:
        negated = False
    name = self.stream.expect('name').value
    while self.stream.current.type == 'dot':
        next(self.stream)
        name += '.' + self.stream.expect('name').value
    dyn_args = dyn_kwargs = None
    kwargs = []
    if self.stream.current.type == 'lparen':
        args, kwargs, dyn_args, dyn_kwargs = self.parse_call_args()
    elif self.stream.current.type in {'name', 'string', 'integer', 'float', 'lparen', 'lbracket', 'lbrace'} and (not self.stream.current.test_any('name:else', 'name:or', 'name:and')):
        if self.stream.current.test('name:is'):
            self.fail('You cannot chain multiple tests with is')
        arg_node = self.parse_primary()
        arg_node = self.parse_postfix(arg_node)
        args = [arg_node]
    else:
        args = []
    node = nodes.Test(node, name, args, kwargs, dyn_args, dyn_kwargs, lineno=token.lineno)
    if negated:
        node = nodes.Not(node, lineno=token.lineno)
    return node