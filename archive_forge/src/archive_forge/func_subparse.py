import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def subparse(self, end_tokens: t.Optional[t.Tuple[str, ...]]=None) -> t.List[nodes.Node]:
    body: t.List[nodes.Node] = []
    data_buffer: t.List[nodes.Node] = []
    add_data = data_buffer.append
    if end_tokens is not None:
        self._end_token_stack.append(end_tokens)

    def flush_data() -> None:
        if data_buffer:
            lineno = data_buffer[0].lineno
            body.append(nodes.Output(data_buffer[:], lineno=lineno))
            del data_buffer[:]
    try:
        while self.stream:
            token = self.stream.current
            if token.type == 'data':
                if token.value:
                    add_data(nodes.TemplateData(token.value, lineno=token.lineno))
                next(self.stream)
            elif token.type == 'variable_begin':
                next(self.stream)
                add_data(self.parse_tuple(with_condexpr=True))
                self.stream.expect('variable_end')
            elif token.type == 'block_begin':
                flush_data()
                next(self.stream)
                if end_tokens is not None and self.stream.current.test_any(*end_tokens):
                    return body
                rv = self.parse_statement()
                if isinstance(rv, list):
                    body.extend(rv)
                else:
                    body.append(rv)
                self.stream.expect('block_end')
            else:
                raise AssertionError('internal parsing error')
        flush_data()
    finally:
        if end_tokens is not None:
            self._end_token_stack.pop()
    return body