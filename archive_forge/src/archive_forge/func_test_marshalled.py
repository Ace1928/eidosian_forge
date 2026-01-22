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
def test_marshalled(self):
    context = mock.Mock()
    self.plugin.prune = mock.Mock()
    self.plugin.marshalled(context)
    self.plugin.prune.assert_called_once_with(context.envelope)
    context.envelope.walk.assert_called_once_with(self.plugin.add_attribute_for_value)