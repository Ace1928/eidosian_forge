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
def test_print_list_with_list_sortby(self):
    Row = collections.namedtuple('Row', ['a', 'b'])
    to_print = [Row(a=4, b=3), Row(a=2, b=1)]
    with CaptureStdout() as cso:
        shell_utils.print_list(to_print, ['a', 'b'], sortby_index=1)
    self.assertEqual('+---+---+\n| a | b |\n+---+---+\n| 2 | 1 |\n| 4 | 3 |\n+---+---+\n', cso.read())