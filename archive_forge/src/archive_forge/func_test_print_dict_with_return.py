import collections
import io
import sys
from unittest import mock
import ddt
from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient.tests.unit import utils as test_utils
from cinderclient import utils
def test_print_dict_with_return(self):
    d = {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'test\rcarriage\n\rreturn'}
    with CaptureStdout() as cso:
        shell_utils.print_dict(d)
    self.assertEqual('+----------+---------------+\n| Property | Value         |\n+----------+---------------+\n| a        | A             |\n| b        | B             |\n| c        | C             |\n| d        | test carriage |\n|          |  return       |\n+----------+---------------+\n', cso.read())