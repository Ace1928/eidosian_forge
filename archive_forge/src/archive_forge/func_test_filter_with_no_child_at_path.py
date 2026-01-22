import http.client as httplib
import io
from unittest import mock
import ddt
import requests
import suds
from oslo_vmware import exceptions
from oslo_vmware import service
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_filter_with_no_child_at_path(self):
    message = mock.Mock(spec=object)
    record = mock.Mock(msg=message)
    self.assertTrue(self.log_filter.filter(record))