import json
from unittest import mock
import uuid
from openstack import exceptions
from openstack.tests.unit import base
def test_http_status(self):
    http_status = 123
    exc = self.assertRaises(exceptions.HttpException, self._do_raise, self.message, http_status=http_status)
    self.assertEqual(self.message, exc.message)
    self.assertEqual(http_status, exc.status_code)