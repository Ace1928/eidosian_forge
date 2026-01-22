from unittest import mock
import uuid
import fixtures
import webob
from keystonemiddleware.tests.unit.audit import base
def test_cadf_event_scoped_to_request_on_error(self):
    middleware = self.create_simple_middleware()
    req = webob.Request.blank('/foo/bar', environ=self.get_environ_header('GET'))
    req.environ['audit.context'] = {}
    self.notifier.notify.side_effect = Exception('error')
    middleware(req)
    self.assertTrue(self.notifier.notify.called)
    req2 = webob.Request.blank('/foo/bar', environ=self.get_environ_header('GET'))
    req2.environ['audit.context'] = {}
    self.notifier.reset_mock()
    middleware._process_response(req2, webob.response.Response())
    self.assertTrue(self.notifier.notify.called)
    self.assertNotEqual(req.environ['cadf_event'].id, self.notifier.notify.call_args_list[0][0][2]['id'])