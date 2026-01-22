import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
def test_pagination_error(self):
    self.manager.api.client.get = mock.Mock(return_value=(None, None))
    self.assertRaises(Exception, self.manager._paginated, self.url, self.response_key)