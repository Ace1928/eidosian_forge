from __future__ import annotations
import glob
import optparse     # pylint: disable=deprecated-module
import os
import os.path
import shlex
import sys
import textwrap
import traceback
from typing import cast, Any, NoReturn
import coverage
from coverage import Coverage
from coverage import env
from coverage.collector import HAS_CTRACER
from coverage.config import CoverageConfig
from coverage.control import DEFAULT_DATAFILE
from coverage.data import combinable_files, debug_data_file
from coverage.debug import info_header, short_stack, write_formatted_info
from coverage.exceptions import _BaseCoverageException, _ExceptionDuringRun, NoSource
from coverage.execfile import PyRunner
from coverage.results import Numbers, should_fail_under
from coverage.version import __url__
def parse_args_ok(self, args: list[str]) -> tuple[bool, optparse.Values | None, list[str]]:
    """Call optparse.parse_args, but return a triple:

        (ok, options, args)

        """
    try:
        options, args = super().parse_args(args)
    except self.OptionParserError:
        return (False, None, [])
    return (True, options, args)