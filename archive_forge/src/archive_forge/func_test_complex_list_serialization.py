import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
def test_complex_list_serialization(self):
    params = {}
    self.connection.build_complex_list_params(params, [('foo', 'bar', 'baz'), ('foo2', 'bar2', 'baz2')], 'ParamName.member', ('One', 'Two', 'Three'))
    self.assertDictEqual({'ParamName.member.1.One': 'foo', 'ParamName.member.1.Two': 'bar', 'ParamName.member.1.Three': 'baz', 'ParamName.member.2.One': 'foo2', 'ParamName.member.2.Two': 'bar2', 'ParamName.member.2.Three': 'baz2'}, params)