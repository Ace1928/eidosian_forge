import os
import unittest
from websocket._url import (
def testProxyFromEnv(self):
    os.environ['http_proxy'] = 'http://localhost/'
    self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', None, None))
    os.environ['http_proxy'] = 'http://localhost:3128/'
    self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', 3128, None))
    os.environ['http_proxy'] = 'http://localhost/'
    os.environ['https_proxy'] = 'http://localhost2/'
    self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', None, None))
    os.environ['http_proxy'] = 'http://localhost:3128/'
    os.environ['https_proxy'] = 'http://localhost2:3128/'
    self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', 3128, None))
    os.environ['http_proxy'] = 'http://localhost/'
    os.environ['https_proxy'] = 'http://localhost2/'
    self.assertEqual(get_proxy_info('echo.websocket.events', True), ('localhost2', None, None))
    os.environ['http_proxy'] = 'http://localhost:3128/'
    os.environ['https_proxy'] = 'http://localhost2:3128/'
    self.assertEqual(get_proxy_info('echo.websocket.events', True), ('localhost2', 3128, None))
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = 'http://localhost2/'
    self.assertEqual(get_proxy_info('echo.websocket.events', True), ('localhost2', None, None))
    self.assertEqual(get_proxy_info('echo.websocket.events', False), (None, 0, None))
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = 'http://localhost2:3128/'
    self.assertEqual(get_proxy_info('echo.websocket.events', True), ('localhost2', 3128, None))
    self.assertEqual(get_proxy_info('echo.websocket.events', False), (None, 0, None))
    os.environ['http_proxy'] = 'http://localhost/'
    os.environ['https_proxy'] = ''
    self.assertEqual(get_proxy_info('echo.websocket.events', True), (None, 0, None))
    self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', None, None))
    os.environ['http_proxy'] = 'http://localhost:3128/'
    os.environ['https_proxy'] = ''
    self.assertEqual(get_proxy_info('echo.websocket.events', True), (None, 0, None))
    self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', 3128, None))
    os.environ['http_proxy'] = 'http://a:b@localhost/'
    self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', None, ('a', 'b')))
    os.environ['http_proxy'] = 'http://a:b@localhost:3128/'
    self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', 3128, ('a', 'b')))
    os.environ['http_proxy'] = 'http://a:b@localhost/'
    os.environ['https_proxy'] = 'http://a:b@localhost2/'
    self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', None, ('a', 'b')))
    os.environ['http_proxy'] = 'http://a:b@localhost:3128/'
    os.environ['https_proxy'] = 'http://a:b@localhost2:3128/'
    self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', 3128, ('a', 'b')))
    os.environ['http_proxy'] = 'http://a:b@localhost/'
    os.environ['https_proxy'] = 'http://a:b@localhost2/'
    self.assertEqual(get_proxy_info('echo.websocket.events', True), ('localhost2', None, ('a', 'b')))
    os.environ['http_proxy'] = 'http://a:b@localhost:3128/'
    os.environ['https_proxy'] = 'http://a:b@localhost2:3128/'
    self.assertEqual(get_proxy_info('echo.websocket.events', True), ('localhost2', 3128, ('a', 'b')))
    os.environ['http_proxy'] = 'http://john%40example.com:P%40SSWORD@localhost:3128/'
    os.environ['https_proxy'] = 'http://john%40example.com:P%40SSWORD@localhost2:3128/'
    self.assertEqual(get_proxy_info('echo.websocket.events', True), ('localhost2', 3128, ('john@example.com', 'P@SSWORD')))
    os.environ['http_proxy'] = 'http://a:b@localhost/'
    os.environ['https_proxy'] = 'http://a:b@localhost2/'
    os.environ['no_proxy'] = 'example1.com,example2.com'
    self.assertEqual(get_proxy_info('example.1.com', True), ('localhost2', None, ('a', 'b')))
    os.environ['http_proxy'] = 'http://a:b@localhost:3128/'
    os.environ['https_proxy'] = 'http://a:b@localhost2:3128/'
    os.environ['no_proxy'] = 'example1.com,example2.com, echo.websocket.events'
    self.assertEqual(get_proxy_info('echo.websocket.events', True), (None, 0, None))
    os.environ['http_proxy'] = 'http://a:b@localhost:3128/'
    os.environ['https_proxy'] = 'http://a:b@localhost2:3128/'
    os.environ['no_proxy'] = 'example1.com,example2.com, .websocket.events'
    self.assertEqual(get_proxy_info('echo.websocket.events', True), (None, 0, None))
    os.environ['http_proxy'] = 'http://a:b@localhost:3128/'
    os.environ['https_proxy'] = 'http://a:b@localhost2:3128/'
    os.environ['no_proxy'] = '127.0.0.0/8, 192.168.0.0/16'
    self.assertEqual(get_proxy_info('127.0.0.1', False), (None, 0, None))
    self.assertEqual(get_proxy_info('192.168.1.1', False), (None, 0, None))