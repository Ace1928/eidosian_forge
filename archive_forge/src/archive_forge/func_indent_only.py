import textwrap
import typing as t
from contextlib import contextmanager
def indent_only(self, text: str) -> str:
    rv = []
    for idx, line in enumerate(text.splitlines()):
        indent = self.initial_indent
        if idx > 0:
            indent = self.subsequent_indent
        rv.append(f'{indent}{line}')
    return '\n'.join(rv)