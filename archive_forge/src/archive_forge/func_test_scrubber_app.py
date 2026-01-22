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
def test_scrubber_app(self):
    """
        test that the glance-scrubber script runs successfully when not in
        daemon mode
        """
    self.cleanup()
    kwargs = self.__dict__.copy()
    self.start_servers(delayed_delete=True, daemon=False, metadata_encryption_key='', **kwargs)
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
    response, content = self._send_http_request(path, 'DELETE')
    self.assertEqual(http.client.NO_CONTENT, response.status)
    image = self._get_pending_delete_image(image['id'])
    self.assertEqual('pending_delete', image['status'])
    time.sleep(self.api_server.scrub_time)
    exe_cmd = '%s -m glance.cmd.scrubber' % sys.executable
    cmd = '%s --config-file %s' % (exe_cmd, self.scrubber_daemon.conf_file_name)
    exitcode, out, err = execute(cmd, raise_error=False)
    self.assertEqual(0, exitcode)
    self.wait_for_scrub(image['id'])
    self.stop_servers()