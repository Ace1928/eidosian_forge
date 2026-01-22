import collections
import collections.abc
import datetime
import functools
import io
import ipaddress
import itertools
import json
from unittest import mock
from xmlrpc import client as xmlrpclib
import netaddr
from oslo_i18n import fixture
from oslotest import base as test_base
from oslo_serialization import jsonutils
def test_loads_unicode(self):
    self.assertIsInstance(jsonutils.loads(b'"foo"'), str)
    self.assertIsInstance(jsonutils.loads('"foo"'), str)
    i18n_str_unicode = '"тест"'
    self.assertIsInstance(jsonutils.loads(i18n_str_unicode), str)
    i18n_str = i18n_str_unicode.encode('utf-8')
    self.assertIsInstance(jsonutils.loads(i18n_str), str)