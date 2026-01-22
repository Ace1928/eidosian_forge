from tests.unit import unittest
from mock import call, Mock, patch, sentinel
import codecs
from boto.glacier.layer1 import Layer1
from boto.glacier.layer2 import Layer2
import boto.glacier.vault
from boto.glacier.vault import Vault
from boto.glacier.vault import Job
from datetime import datetime, tzinfo, timedelta
def test_create_archive_writer(self):
    self.mock_layer1.initiate_multipart_upload.return_value = {'UploadId': 'UPLOADID'}
    writer = self.vault.create_archive_writer(description='stuff')
    self.mock_layer1.initiate_multipart_upload.assert_called_with('examplevault', self.vault.DefaultPartSize, 'stuff')
    self.assertEqual(writer.vault, self.vault)
    self.assertEqual(writer.upload_id, 'UPLOADID')