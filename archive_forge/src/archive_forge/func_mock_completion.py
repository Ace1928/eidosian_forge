import json
import os
from unittest import mock
import fixtures
import requests
from requests_mock.contrib import fixture as requests_mock_fixture
import testtools
def mock_completion(self):
    patcher = mock.patch('cinderclient.base.Manager.write_to_completion_cache')
    patcher.start()
    self.addCleanup(patcher.stop)
    patcher = mock.patch('cinderclient.base.Manager.completion_cache')
    patcher.start()
    self.addCleanup(patcher.stop)