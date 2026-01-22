import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_call_block(self) -> nodes.CallBlock:
    node = nodes.CallBlock(lineno=next(self.stream).lineno)
    if self.stream.current.type == 'lparen':
        self.parse_signature(node)
    else:
        node.args = []
        node.defaults = []
    call_node = self.parse_expression()
    if not isinstance(call_node, nodes.Call):
        self.fail('expected call', node.lineno)
    node.call = call_node
    node.body = self.parse_statements(('name:endcall',), drop_needle=True)
    return node