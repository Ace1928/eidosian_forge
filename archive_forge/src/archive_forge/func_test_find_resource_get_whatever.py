import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_find_resource_get_whatever(self):
    self.manager.get = mock.Mock(return_value=self.expected)
    result = utils.find_resource(self.manager, 'whatever')
    self.assertEqual(self.expected, result)
    self.manager.get.assert_called_with('whatever')