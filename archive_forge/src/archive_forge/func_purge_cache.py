import os
import textwrap
from optparse import Values
from typing import Any, List
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.exceptions import CommandError, PipError
from pip._internal.utils import filesystem
from pip._internal.utils.logging import getLogger
def purge_cache(self, options: Values, args: List[Any]) -> None:
    if args:
        raise CommandError('Too many arguments')
    return self.remove_cache_items(options, ['*'])