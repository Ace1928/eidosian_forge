import os
import unittest
from websocket._url import (
def testIpAddressInRange(self):
    self.assertTrue(_is_no_proxy_host('127.0.0.1', ['127.0.0.0/8']))
    self.assertTrue(_is_no_proxy_host('127.0.0.2', ['127.0.0.0/8']))
    self.assertFalse(_is_no_proxy_host('127.1.0.1', ['127.0.0.0/24']))
    os.environ['no_proxy'] = '127.0.0.0/8'
    self.assertTrue(_is_no_proxy_host('127.0.0.1', None))
    self.assertTrue(_is_no_proxy_host('127.0.0.2', None))
    os.environ['no_proxy'] = '127.0.0.0/24'
    self.assertFalse(_is_no_proxy_host('127.1.0.1', None))