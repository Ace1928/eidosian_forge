import unittest
from mock import Mock, call
from .watch import Watch
def test_watch_stream_loop(self):
    w = Watch(float)
    fake_resp = Mock()
    fake_resp.close = Mock()
    fake_resp.release_conn = Mock()
    fake_resp.read_chunked = Mock(return_value=['{"type": "ADDED", "object": 1}\n'])
    fake_api = Mock()
    fake_api.get_namespaces = Mock(return_value=fake_resp)
    fake_api.get_namespaces.__doc__ = ':return: V1NamespaceList'
    count = 0
    for e in w.stream(fake_api.get_namespaces, timeout_seconds=1):
        count = count + 1
    self.assertEqual(count, 1)
    for e in w.stream(fake_api.get_namespaces):
        count = count + 1
        if count == 2:
            w.stop()
    self.assertEqual(count, 2)
    self.assertEqual(fake_api.get_namespaces.call_count, 2)
    self.assertEqual(fake_resp.read_chunked.call_count, 2)
    self.assertEqual(fake_resp.close.call_count, 2)
    self.assertEqual(fake_resp.release_conn.call_count, 2)