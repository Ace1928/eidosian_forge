import os
import textwrap
from optparse import Values
from typing import Any, List
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.exceptions import CommandError, PipError
from pip._internal.utils import filesystem
from pip._internal.utils.logging import getLogger
def list_cache_items(self, options: Values, args: List[Any]) -> None:
    if len(args) > 1:
        raise CommandError('Too many arguments')
    if args:
        pattern = args[0]
    else:
        pattern = '*'
    files = self._find_wheels(options, pattern)
    if options.list_format == 'human':
        self.format_for_human(files)
    else:
        self.format_for_abspath(files)