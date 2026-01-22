import http.client
import os
import sys
import time
import httplib2
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils.fixture import uuidsentinel as uuids
from glance import context
import glance.db as db_api
from glance.tests import functional
from glance.tests.utils import execute
def test_scrubber_restore_active_image_raise_error(self):
    self.cleanup()
    self.start_servers(delayed_delete=True, daemon=False, metadata_encryption_key='')
    path = 'http://%s:%d/v2/images' % ('127.0.0.1', self.api_port)
    response, content = self._send_create_image_http_request(path)
    self.assertEqual(http.client.CREATED, response.status)
    image = jsonutils.loads(content)
    self.assertEqual('queued', image['status'])
    file_path = '%s/%s/file' % (path, image['id'])
    response, content = self._send_upload_image_http_request(file_path, body='XXX')
    self.assertEqual(http.client.NO_CONTENT, response.status)
    path = '%s/%s' % (path, image['id'])
    response, content = self._send_http_request(path, 'GET')
    image = jsonutils.loads(content)
    self.assertEqual('active', image['status'])

    def _test_content():
        exe_cmd = '%s -m glance.cmd.scrubber' % sys.executable
        cmd = '%s --config-file %s --restore %s' % (exe_cmd, self.scrubber_daemon.conf_file_name, image['id'])
        return execute(cmd, raise_error=False)
    exitcode, out, err = self.wait_for_scrubber_shutdown(_test_content)
    self.assertEqual(1, exitcode)
    self.assertIn('cannot restore the image from active to active (wanted from_state=pending_delete)', str(err))
    self.stop_servers()