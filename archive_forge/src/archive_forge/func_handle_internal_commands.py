import click
import os
import shlex
import sys
from collections import defaultdict
from .exceptions import CommandLineParserError, ExitReplException
def handle_internal_commands(command):
    """
    Run repl-internal commands.

    Repl-internal commands are all commands starting with ":".
    """
    if command.startswith(':'):
        target = _get_registered_target(command[1:], default=None)
        if target:
            return target()