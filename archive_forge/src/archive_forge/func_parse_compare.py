from typing import Iterator, List, Optional
from jinja2 import nodes
from jinja2.environment import Environment
from jinja2.nodes import Node
from jinja2.parser import Parser
def parse_compare(self) -> Node:
    node: Node
    token = self.stream.current
    if token.type == 'name':
        if token.value in ('true', 'false', 'True', 'False'):
            node = nodes.Const(token.value in ('true', 'True'), lineno=token.lineno)
        elif token.value in ('none', 'None'):
            node = nodes.Const(None, lineno=token.lineno)
        else:
            node = nodes.Name(token.value, 'load', lineno=token.lineno)
        next(self.stream)
    elif token.type == 'lparen':
        next(self.stream)
        node = self.parse_expression()
        self.stream.expect('rparen')
    else:
        self.fail("unexpected token '%s'" % (token,), token.lineno)
    return node