import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_print(self) -> nodes.Output:
    node = nodes.Output(lineno=next(self.stream).lineno)
    node.nodes = []
    while self.stream.current.type != 'block_end':
        if node.nodes:
            self.stream.expect('comma')
        node.nodes.append(self.parse_expression())
    return node