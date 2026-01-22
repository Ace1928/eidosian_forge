import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_find_resource_list_forbidden(self):
    self.manager.get = mock.Mock(side_effect=Exception('Boom!'))
    self.manager.find = mock.Mock(side_effect=Exception('Boom!'))
    self.manager.list = mock.Mock(side_effect=exceptions.Forbidden(403))
    self.assertRaises(exceptions.Forbidden, utils.find_resource, self.manager, self.name)
    self.manager.list.assert_called_with()