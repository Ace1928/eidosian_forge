import json
import copy
import tempfile
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.layer1 import Layer1
from boto.compat import six
def test_initiate_archive_job(self):
    content = {u'Type': u'archive-retrieval', u'ArchiveId': u'AAABZpJrTyioDC_HsOmHae8EZp_uBSJr6cnGOLKp_XJCl-Q', u'Description': u'Test Archive', u'SNSTopic': u'Topic', u'JobId': None, u'Location': None, u'RequestId': None}
    self.set_http_response(status_code=202, header=self.json_header, body=json.dumps(content).encode('utf-8'))
    api_response = self.service_connection.initiate_job(self.vault_name, self.job_content)
    self.assertDictEqual(content, api_response)