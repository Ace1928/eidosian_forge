from tests.unit import unittest
from mock import call, Mock, patch, sentinel
import codecs
from boto.glacier.layer1 import Layer1
from boto.glacier.layer2 import Layer2
import boto.glacier.vault
from boto.glacier.vault import Vault
from boto.glacier.vault import Job
from datetime import datetime, tzinfo, timedelta
def test_delete_vault(self):
    self.vault.delete_archive('archive')
    self.mock_layer1.delete_archive.assert_called_with('examplevault', 'archive')