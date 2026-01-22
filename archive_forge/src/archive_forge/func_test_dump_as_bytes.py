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
def test_dump_as_bytes(self):
    self.assertEqual(b'{"a": "b"}', jsonutils.dump_as_bytes({'a': 'b'}))