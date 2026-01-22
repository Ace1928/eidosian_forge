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
def test_print_list_with_return(self):
    Row = collections.namedtuple('Row', ['a', 'b'])
    to_print = [Row(a=3, b='a\r'), Row(a=1, b='c\rd')]
    with CaptureStdout() as cso:
        shell_utils.print_list(to_print, ['a', 'b'])
    self.assertEqual('+---+-----+\n| a | b   |\n+---+-----+\n| 1 | c d |\n| 3 | a   |\n+---+-----+\n', cso.read())