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
def test_print_list_with_generator(self):
    Row = collections.namedtuple('Row', ['a', 'b'])

    def gen_rows():
        for row in [Row(a=1, b=2), Row(a=3, b=4)]:
            yield row
    with CaptureStdout() as cso:
        shell_utils.print_list(gen_rows(), ['a', 'b'])
    self.assertEqual('+---+---+\n| a | b |\n+---+---+\n| 1 | 2 |\n| 3 | 4 |\n+---+---+\n', cso.read())