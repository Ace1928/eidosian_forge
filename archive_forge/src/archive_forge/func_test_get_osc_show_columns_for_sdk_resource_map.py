import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_get_osc_show_columns_for_sdk_resource_map(self):
    self._test_get_osc_show_columns_for_sdk_resource({'foo': 'foo1'}, {'foo': 'foo_map'}, ('foo_map',), ('foo',))