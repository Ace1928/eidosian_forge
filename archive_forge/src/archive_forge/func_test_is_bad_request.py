from unittest import mock
from openstack import exceptions
from heat.engine.clients.os import senlin as senlin_plugin
from heat.tests import common
from heat.tests import utils
def test_is_bad_request(self):
    self.assertTrue(self.plugin.is_bad_request(exceptions.HttpException(http_status=400)))
    self.assertFalse(self.plugin.is_bad_request(Exception))
    self.assertFalse(self.plugin.is_bad_request(exceptions.HttpException(http_status=404)))