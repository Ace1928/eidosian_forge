from unittest import mock
import uuid
import fixtures
import webob
from keystonemiddleware.tests.unit.audit import base
def test_cadf_event_scoped_to_request(self):
    app = self.create_simple_app()
    resp = app.get('/foo/bar', extra_environ=self.get_environ_header())
    self.assertIsNotNone(resp.request.environ.get('cadf_event'))
    self.assertEqual(self.notifier.calls[0].payload['id'], self.notifier.calls[1].payload['id'])