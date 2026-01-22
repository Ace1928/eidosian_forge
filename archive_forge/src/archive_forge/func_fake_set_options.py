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
def fake_set_options(*args, **kwargs):
    headers = kwargs['soapheaders']
    self.assertEqual(2, len(headers))
    self.assertEqual('vc-session-cookie', headers[0].getText())
    self.assertEqual('fira-12345', headers[1].getText())