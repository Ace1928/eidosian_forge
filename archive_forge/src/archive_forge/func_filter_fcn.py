import enum
import glob
import logging
import math
import os
import operator
import re
import subprocess
import sys
from io import StringIO
from unittest import *
import unittest as _unittest
import pytest as pytest
from pyomo.common.collections import Mapping, Sequence
from pyomo.common.dependencies import attempt_import, check_min_version
from pyomo.common.errors import InvalidValueError
from pyomo.common.fileutils import import_file
from pyomo.common.log import LoggingIntercept, pyomo_formatter
from pyomo.common.tee import capture_output
from unittest import mock
def filter_fcn(self, line):
    """
        Ignore certain text when comparing output with baseline
        """
    for field in ('[', 'password:', 'http:', 'Job ', 'Importing module', 'Function', 'File', 'Matplotlib', 'Memory:', '-------', '=======', '    ^'):
        if line.startswith(field):
            return True
    for field in ('Total CPU', 'Ipopt', 'license', 'time:', 'Time:', 'with format cpxlp', 'usermodel = <module', 'execution time=', 'Solver results file:', 'TokenServer', 'function calls', 'List reduced', '.py:', ' {built-in method', ' {method', ' {pyomo.core.expr.numvalue.as_numeric}'):
        if field in line:
            return True
    return False