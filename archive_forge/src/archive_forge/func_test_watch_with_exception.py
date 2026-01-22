import unittest
from mock import Mock, call
from .watch import Watch
def test_watch_with_exception(self):
    fake_resp = Mock()
    fake_resp.close = Mock()
    fake_resp.release_conn = Mock()
    fake_resp.read_chunked = Mock(side_effect=KeyError('expected'))
    fake_api = Mock()
    fake_api.get_thing = Mock(return_value=fake_resp)
    w = Watch()
    try:
        for _ in w.stream(fake_api.get_thing):
            self.fail(self, 'Should fail on exception.')
    except KeyError:
        pass
    fake_api.get_thing.assert_called_once_with(_preload_content=False, watch=True)
    fake_resp.read_chunked.assert_called_once_with(decode_content=False)
    fake_resp.close.assert_called_once()
    fake_resp.release_conn.assert_called_once()