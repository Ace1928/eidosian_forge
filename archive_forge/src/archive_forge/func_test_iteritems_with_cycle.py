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
def test_iteritems_with_cycle(self):

    class IterItemsClass(object):

        def __init__(self):
            self.data = dict(a=1, b=2, c=3)
            self.index = 0

        def iteritems(self):
            return self.data.items()
    x = IterItemsClass()
    x2 = IterItemsClass()
    x.data['other'] = x2
    x2.data['other'] = x
    jsonutils.to_primitive(x)