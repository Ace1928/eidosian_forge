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
def test_DateTime(self):
    x = xmlrpclib.DateTime()
    x.decode('19710203T04:05:06')
    self.assertEqual('1971-02-03T04:05:06.000000', jsonutils.to_primitive(x))