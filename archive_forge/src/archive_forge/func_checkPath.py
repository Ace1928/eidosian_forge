import ast
import os
import platform
import re
import sys
from pyflakes import checker, __version__
from pyflakes import reporter as modReporter
def checkPath(filename, reporter=None):
    """
    Check the given path, printing out any warnings detected.

    @param reporter: A L{Reporter} instance, where errors and warnings will be
        reported.

    @return: the number of warnings printed
    """
    if reporter is None:
        reporter = modReporter._makeDefaultReporter()
    try:
        with open(filename, 'rb') as f:
            codestr = f.read()
    except OSError as e:
        reporter.unexpectedError(filename, e.args[1])
        return 1
    return check(codestr, filename, reporter)