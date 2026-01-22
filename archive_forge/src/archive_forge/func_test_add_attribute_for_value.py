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
@ddt.data(('value', 'foo', 'string'), ('removeKey', '1', 'int'), ('removeKey', 'foo', 'string'))
@ddt.unpack
def test_add_attribute_for_value(self, name, text, expected_xsd_type):
    node = mock.Mock()
    node.name = name
    node.text = text
    self.plugin.add_attribute_for_value(node)
    node.set.assert_called_once_with('xsi:type', 'xsd:%s' % expected_xsd_type)