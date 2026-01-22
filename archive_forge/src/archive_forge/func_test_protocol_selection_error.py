import unittest
from wsme import WSRoot
import wsme.protocol
import wsme.rest.protocol
from wsme.root import default_prepare_response_body
from webob import Request
def test_protocol_selection_error(self):

    class P(wsme.protocol.Protocol):
        name = 'test'

        def accept(self, r):
            raise Exception('test')
    root = WSRoot()
    root.addprotocol(P())
    from webob import Request
    req = Request.blank('/test?check=a&check=b&name=Bob')
    res = root._handle_request(req)
    assert res.status_int == 500
    assert res.content_type == 'text/plain'
    assert res.text == 'Unexpected error while selecting protocol: test', req.text