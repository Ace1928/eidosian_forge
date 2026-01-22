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
@classmethod
def gather_tests(cls, test_dirs):
    sh_test_tuples = cls._find_tests(test_dirs, '*.sh')
    py_test_tuples = cls._find_tests(test_dirs, '*.py')
    sh_files = set(map(operator.itemgetter(1), sh_test_tuples))
    py_test_tuples = list(filter(lambda t: t[1][:-3] + '.sh' not in sh_files, py_test_tuples))
    return (py_test_tuples, sh_test_tuples)