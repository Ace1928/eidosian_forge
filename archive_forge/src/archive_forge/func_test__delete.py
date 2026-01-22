import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
def test__delete(self):
    resp_ = 'test-resp'
    body_ = 'test-body'
    manager = self.get_mock_mng_api_client()
    manager.api.client.delete = mock.Mock(return_value=(resp_, body_))
    manager._delete('test-url')
    pass