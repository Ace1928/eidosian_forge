import os
import unittest
from websocket._url import (
def testHostnameMatchDomain(self):
    self.assertTrue(_is_no_proxy_host('any.websocket.org', ['.websocket.org']))
    self.assertTrue(_is_no_proxy_host('my.other.websocket.org', ['.websocket.org']))
    self.assertTrue(_is_no_proxy_host('any.websocket.org', ['my.websocket.org', '.websocket.org']))
    self.assertFalse(_is_no_proxy_host('any.websocket.com', ['.websocket.org']))
    os.environ['no_proxy'] = '.websocket.org'
    self.assertTrue(_is_no_proxy_host('any.websocket.org', None))
    self.assertTrue(_is_no_proxy_host('my.other.websocket.org', None))
    self.assertFalse(_is_no_proxy_host('any.websocket.com', None))
    os.environ['no_proxy'] = 'my.websocket.org, .websocket.org'
    self.assertTrue(_is_no_proxy_host('any.websocket.org', None))