from __future__ import with_statement
import click
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from ._completer import ClickCompleter
from .exceptions import ClickExit  # type: ignore[attr-defined]
from .exceptions import CommandLineParserError, ExitReplException, InvalidGroupFormat
from .utils import _execute_internal_and_sys_cmds
def register_repl(group, name='repl'):
    """Register :func:`repl()` as sub-command *name* of *group*."""
    group.command(name=name)(click.pass_context(repl))