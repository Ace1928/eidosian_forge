import datetime
from unittest import mock
from testtools import matchers
from heat.engine.clients.os import swift
from heat.tests import common
from heat.tests import utils
def test_is_valid_temp_url_path(self):
    valids = ['/v1/AUTH_demo/c/o', '/v1/AUTH_demo/c/o/', '/v1/TEST_demo/c/o', '/v1/AUTH_demo/c/pseudo_folder/o']
    for url in valids:
        self.assertTrue(self.swift_plugin.is_valid_temp_url_path(url))
    invalids = ['/v2/AUTH_demo/c/o', '/v1/AUTH_demo/c//', '/v1/AUTH_demo/c/', '/AUTH_demo/c//', '//AUTH_demo/c/o', '//v1/AUTH_demo/c/o', '/v1/AUTH_demo/o', '/v1/AUTH_demo//o', '/v1/AUTH_d3mo//o', '/v1//c/o', '/v1/c/o']
    for url in invalids:
        self.assertFalse(self.swift_plugin.is_valid_temp_url_path(url))