import os
import base64
from libcloud.utils.py3 import b
from libcloud.common.kubernetes import (
def test_bearer_token_auth(self):
    driver = self.driver_cls(ex_token_bearer_auth=True, key='foobar')
    self.assertEqual(driver.connectionCls, KubernetesTokenAuthConnection)
    self.assertEqual(driver.connection.key, 'foobar')
    headers = driver.connection.add_default_headers({})
    self.assertEqual(headers['Content-Type'], 'application/json')
    self.assertEqual(headers['Authorization'], 'Bearer %s' % 'foobar')