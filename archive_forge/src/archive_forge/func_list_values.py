import logging
import os
import subprocess
from optparse import Values
from typing import Any, List, Optional
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.configuration import (
from pip._internal.exceptions import PipError
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import get_prog, write_output
def list_values(self, options: Values, args: List[str]) -> None:
    self._get_n_args(args, 'list', n=0)
    for key, value in sorted(self.configuration.items()):
        write_output('%s=%r', key, value)