import os
import base64
from libcloud.utils.py3 import b
from libcloud.common.kubernetes import (
def test_cert_auth(self):
    expected_msg = 'Both key and certificate files are needed'
    self.assertRaisesRegex(ValueError, expected_msg, self.driver_cls, key_file=KEY_FILE, ca_cert=CA_CERT_FILE)
    expected_msg = 'Both key and certificate files are needed'
    self.assertRaisesRegex(ValueError, expected_msg, self.driver_cls, cert_file=CERT_FILE, ca_cert=CA_CERT_FILE)
    driver = self.driver_cls(key_file=KEY_FILE, cert_file=CERT_FILE, ca_cert=CA_CERT_FILE)
    self.assertEqual(driver.connectionCls, KubernetesTLSAuthConnection)
    self.assertEqual(driver.connection.key_file, KEY_FILE)
    self.assertEqual(driver.connection.cert_file, CERT_FILE)
    self.assertEqual(driver.connection.connection.ca_cert, CA_CERT_FILE)
    headers = driver.connection.add_default_headers({})
    self.assertEqual(headers['Content-Type'], 'application/json')
    driver = self.driver_cls(key_file=KEY_FILE, cert_file=CERT_FILE, ca_cert=None)
    self.assertEqual(driver.connectionCls, KubernetesTLSAuthConnection)
    self.assertEqual(driver.connection.key_file, KEY_FILE)
    self.assertEqual(driver.connection.cert_file, CERT_FILE)
    self.assertEqual(driver.connection.connection.ca_cert, False)
    headers = driver.connection.add_default_headers({})
    self.assertEqual(headers['Content-Type'], 'application/json')