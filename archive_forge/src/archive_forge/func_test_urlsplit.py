import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_urlsplit(self):
    result = netutils.urlsplit('rpc://myhost?someparam#somefragment')
    self.assertEqual(result.scheme, 'rpc')
    self.assertEqual(result.netloc, 'myhost')
    self.assertEqual(result.path, '')
    self.assertEqual(result.query, 'someparam')
    self.assertEqual(result.fragment, 'somefragment')
    result = netutils.urlsplit('rpc://myhost/mypath?someparam#somefragment', allow_fragments=False)
    self.assertEqual(result.scheme, 'rpc')
    self.assertEqual(result.netloc, 'myhost')
    self.assertEqual(result.path, '/mypath')
    self.assertEqual(result.query, 'someparam#somefragment')
    self.assertEqual(result.fragment, '')
    result = netutils.urlsplit('rpc://user:pass@myhost/mypath?someparam#somefragment', allow_fragments=False)
    self.assertEqual(result.scheme, 'rpc')
    self.assertEqual(result.netloc, 'user:pass@myhost')
    self.assertEqual(result.path, '/mypath')
    self.assertEqual(result.query, 'someparam#somefragment')
    self.assertEqual(result.fragment, '')