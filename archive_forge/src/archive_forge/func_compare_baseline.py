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
def compare_baseline(self, test_output, baseline, abstol=1e-06, reltol=None):
    out_filtered = self.filter_file_contents(test_output.strip().split('\n'), abstol)
    base_filtered = self.filter_file_contents(baseline.strip().split('\n'), abstol)
    try:
        self.assertStructuredAlmostEqual(out_filtered, base_filtered, abstol=abstol, reltol=reltol, allow_second_superset=False)
        return True
    except self.failureException:
        print('---------------------------------')
        print('BASELINE FILE')
        print('---------------------------------')
        print(baseline)
        print('=================================')
        print('---------------------------------')
        print('TEST OUTPUT FILE')
        print('---------------------------------')
        print(test_output)
        raise