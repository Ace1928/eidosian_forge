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
def test_fallback_typeerror_IO_object(self):
    obj = io.IOBase
    ret = jsonutils.to_primitive(obj)
    self.assertEqual(str(obj), ret)
    ret = jsonutils.to_primitive(obj, fallback=lambda _: 'fallback')
    self.assertEqual('fallback', ret)