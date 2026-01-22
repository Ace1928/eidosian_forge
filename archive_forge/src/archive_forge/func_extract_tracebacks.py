import sys
import os
import io
import optparse
import re
import unittest
from pprint import pformat
from .test_machinery import PygameTestLoader
def extract_tracebacks(output):
    """from test runner output return the tracebacks."""
    verbose_mode = ' ...' in output
    if verbose_mode:
        if 'ERROR' in output or 'FAILURE' in output:
            return '\n\n=='.join(output.split('\n\n==')[1:])
    else:
        dots = DOTS.search(output).group(1)
        if 'E' in dots or 'F' in dots:
            return output[len(dots) + 1:].split(RAN_TESTS_DIV)[0]
    return ''