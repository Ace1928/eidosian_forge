import os
import textwrap
from optparse import Values
from typing import Any, List
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.exceptions import CommandError, PipError
from pip._internal.utils import filesystem
from pip._internal.utils.logging import getLogger
def format_for_human(self, files: List[str]) -> None:
    if not files:
        logger.info('No locally built wheels cached.')
        return
    results = []
    for filename in files:
        wheel = os.path.basename(filename)
        size = filesystem.format_file_size(filename)
        results.append(f' - {wheel} ({size})')
    logger.info('Cache contents:\n')
    logger.info('\n'.join(sorted(results)))