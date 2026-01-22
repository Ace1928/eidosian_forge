import sys
import autocommand
from jaraco.text import Stripper

    Strip any common prefix from stdin.

    >>> import io, pytest
    >>> getfixture('monkeypatch').setattr('sys.stdin', io.StringIO('abcdef\nabc123'))
    >>> strip_prefix()
    def
    123
    