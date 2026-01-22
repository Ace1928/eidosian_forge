import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
def test__get_with_response_key(self):
    manager = self._get_mock()
    response_key = 'response_key'
    body_ = {response_key: 'test-resp-key-body'}
    url_ = 'test_url_get'
    manager.api.client.get = mock.Mock(return_value=(url_, body_))
    r = manager._get(url=url_, response_key=response_key)
    self.assertEqual(body_[response_key], r)