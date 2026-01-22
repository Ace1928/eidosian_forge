import re
import sys
import datetime
import unittest
import traceback
from unittest.mock import patch, mock_open
from libcloud.test import MockHttp
from libcloud.utils.py3 import ET, PY2, b, httplib, assertRaisesRegex
from libcloud.compute.base import Node, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import VCLOUD_PARAMS
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vcloud import (
def print_parameterized_failure(names_values):
    """
    Print failure information for a failed, parameterized test.
    This includes a traceback and parameter values.

    :param names_values: Name, value pairs for parameters of test at failure
    :type names_values: ``list`` of (``str``,  ``Any``)

    :return: None
    :rtype: ``None``
    """
    formatted_names_values = ('    {name}={value}'.format(name=name, value=value) for name, value in names_values)
    traceback.print_exc()
    print('Data values:\n{values}\n'.format(values='\n'.join(formatted_names_values)), file=sys.stderr)