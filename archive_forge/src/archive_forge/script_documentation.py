import doctest
import errno
import glob
import logging
import os
import shlex
import sys
import textwrap
from .. import osutils, tests, trace
from ..tests import ui_testing
Run a shell-like script as a test.

        :param test_case: A TestCase instance that should provide the fail(),
            assertEqualDiff and _run_bzr_core() methods as well as a 'test_dir'
            attribute used as a jail root.

        :param text: A shell-like script (see _script_to_commands for syntax).

        :param null_output_matches_anything: For commands with no specified
            output, ignore any output that does happen, including output on
            standard error.
        