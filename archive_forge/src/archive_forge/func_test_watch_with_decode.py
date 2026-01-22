import unittest
from mock import Mock, call
from .watch import Watch
def test_watch_with_decode(self):
    fake_resp = Mock()
    fake_resp.close = Mock()
    fake_resp.release_conn = Mock()
    fake_resp.read_chunked = Mock(return_value=['{"type": "ADDED", "object": {"metadata": {"name": "test1","resourceVersion": "1"}, "spec": {}, "status": {}}}\n', '{"type": "ADDED", "object": {"metadata": {"name": "test2","resourceVersion": "2"}, "spec": {}, "sta', 'tus": {}}}\n{"type": "ADDED", "object": {"metadata": {"name": "test3","resourceVersion": "3"}, "spec": {}, "status": {}}}\n', 'should_not_happened\n'])
    fake_api = Mock()
    fake_api.get_namespaces = Mock(return_value=fake_resp)
    fake_api.get_namespaces.__doc__ = ':return: V1NamespaceList'
    w = Watch()
    count = 1
    for e in w.stream(fake_api.get_namespaces):
        self.assertEqual('ADDED', e['type'])
        self.assertEqual('test%d' % count, e['object'].metadata.name)
        self.assertEqual('%d' % count, e['object'].metadata.resource_version)
        self.assertEqual('%d' % count, w.resource_version)
        count += 1
        if count == 4:
            w.stop()
    fake_api.get_namespaces.assert_called_once_with(_preload_content=False, watch=True)
    fake_resp.read_chunked.assert_called_once_with(decode_content=False)
    fake_resp.close.assert_called_once()
    fake_resp.release_conn.assert_called_once()