import datetime
from unittest import mock
from testtools import matchers
from heat.engine.clients.os import swift
from heat.tests import common
from heat.tests import utils
def test_get_temp_url_no_account_key(self):
    self.swift_client.url = 'http://fake-host.com:8080/v1/AUTH_demo'
    head_account = {}

    def post_account(data):
        head_account.update(data)
    self.swift_client.head_account = mock.Mock(return_value=head_account)
    self.swift_client.post_account = post_account
    container_name = '1234'
    stack_name = 'test'
    handle_name = 'foo'
    obj_name = '%s-%s' % (stack_name, handle_name)
    self.assertNotIn('x-account-meta-temp-url-key', head_account)
    self.swift_plugin.get_temp_url(container_name, obj_name)
    self.assertIn('x-account-meta-temp-url-key', head_account)