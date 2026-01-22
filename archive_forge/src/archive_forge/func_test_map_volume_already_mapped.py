import json
import os
from unittest import mock
import requests
from os_brick import exception
from os_brick.initiator.connectors import scaleio
from os_brick.tests.initiator import test_connector
def test_map_volume_already_mapped(self):
    """Ignore REST API failure for volume already mapped"""
    self.mock_calls[self.action_format.format('addMappedSdc')] = self.MockHTTPSResponse(dict(errorCode=self.connector.VOLUME_ALREADY_MAPPED_ERROR, message='Test error map volume'), 500)
    self.test_connect_volume()