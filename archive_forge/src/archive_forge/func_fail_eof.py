import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def fail_eof(self, end_tokens: t.Optional[t.Tuple[str, ...]]=None, lineno: t.Optional[int]=None) -> 'te.NoReturn':
    """Like fail_unknown_tag but for end of template situations."""
    stack = list(self._end_token_stack)
    if end_tokens is not None:
        stack.append(end_tokens)
    self._fail_ut_eof(None, stack, lineno)