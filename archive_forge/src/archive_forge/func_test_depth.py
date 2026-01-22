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
def test_depth(self):

    class LevelsGenerator(object):

        def __init__(self, levels):
            self._levels = levels

        def iteritems(self):
            if self._levels == 0:
                return iter([])
            else:
                return iter([(0, LevelsGenerator(self._levels - 1))])
    l4_obj = LevelsGenerator(4)
    json_l2 = {0: {0: None}}
    json_l3 = {0: {0: {0: None}}}
    json_l4 = {0: {0: {0: {0: None}}}}
    ret = jsonutils.to_primitive(l4_obj, max_depth=2)
    self.assertEqual(json_l2, ret)
    ret = jsonutils.to_primitive(l4_obj, max_depth=3)
    self.assertEqual(json_l3, ret)
    ret = jsonutils.to_primitive(l4_obj, max_depth=4)
    self.assertEqual(json_l4, ret)