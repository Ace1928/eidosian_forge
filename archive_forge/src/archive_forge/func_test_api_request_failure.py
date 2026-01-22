from unittest import mock
import uuid
import fixtures
import webob
from keystonemiddleware.tests.unit.audit import base
def test_api_request_failure(self):

    class CustomException(Exception):
        pass

    def cb(req):
        raise CustomException('It happens!')
    try:
        self.create_app(cb).get('/foo/bar', extra_environ=self.get_environ_header())
        self.fail('Application exception has not been re-raised')
    except CustomException:
        pass
    call_args = self.notifier.notify.call_args_list[0][0]
    self.assertEqual('audit.http.request', call_args[1])
    self.assertEqual('/foo/bar', call_args[2]['requestPath'])
    self.assertEqual('pending', call_args[2]['outcome'])
    self.assertNotIn('reporterchain', call_args[2])
    call_args = self.notifier.notify.call_args_list[1][0]
    self.assertEqual('audit.http.response', call_args[1])
    self.assertEqual('/foo/bar', call_args[2]['requestPath'])
    self.assertEqual('unknown', call_args[2]['outcome'])
    self.assertIn('reporterchain', call_args[2])