import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_calculate_header_and_attrs_without_column_arg(self):
    self._test_calculate_header_and_attrs([], ('ID', 'Name', 'Fixed IP Addresses'), ('id', 'name', 'fixed_ips'))