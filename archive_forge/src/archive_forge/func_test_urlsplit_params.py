import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_urlsplit_params(self):
    test_url = 'http://localhost/?a=b&c=d'
    result = netutils.urlsplit(test_url)
    self.assertEqual({'a': 'b', 'c': 'd'}, result.params())
    self.assertEqual({'a': 'b', 'c': 'd'}, result.params(collapse=False))
    test_url = 'http://localhost/?a=b&a=c&a=d'
    result = netutils.urlsplit(test_url)
    self.assertEqual({'a': 'd'}, result.params())
    self.assertEqual({'a': ['b', 'c', 'd']}, result.params(collapse=False))
    test_url = 'http://localhost'
    result = netutils.urlsplit(test_url)
    self.assertEqual({}, result.params())
    test_url = 'http://localhost?'
    result = netutils.urlsplit(test_url)
    self.assertEqual({}, result.params())