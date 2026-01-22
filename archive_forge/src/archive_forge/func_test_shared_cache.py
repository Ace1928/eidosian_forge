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
@mock.patch('suds.reader.DefinitionsReader.open')
@mock.patch('suds.reader.DocumentReader.download', create=True)
def test_shared_cache(self, mock_reader, mock_open):
    cache1 = service.Service().client.options.cache
    cache2 = service.Service().client.options.cache
    self.assertIs(cache1, cache2)