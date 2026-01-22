import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
def test_pagination_next(self):
    resp = self.manager._paginated(self.url, self.response_key, limit=self.limit, marker=self.marker)
    self.manager.api.client.get.assert_called_with(self.next_url)
    self.assertEqual('p3', resp[0].foo)
    self.assertEqual('p4', resp[1].foo)
    self.assertIsNone(resp.next)
    self.assertEqual([], resp.links)
    self.assertIsInstance(resp, common.Paginated)