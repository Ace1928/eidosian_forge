from tests.unit import unittest
from mock import call, Mock, patch, sentinel
import codecs
from boto.glacier.layer1 import Layer1
from boto.glacier.layer2 import Layer2
import boto.glacier.vault
from boto.glacier.vault import Vault
from boto.glacier.vault import Job
from datetime import datetime, tzinfo, timedelta
def test_create_vault(self):
    self.mock_layer1.describe_vault.return_value = FIXTURE_VAULT
    self.layer2.create_vault('My Vault')
    self.mock_layer1.create_vault.assert_called_with('My Vault')