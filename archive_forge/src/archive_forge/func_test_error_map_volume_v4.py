import json
import os
from unittest import mock
import requests
from os_brick import exception
from os_brick.initiator.connectors import scaleio
from os_brick.tests.initiator import test_connector
def test_error_map_volume_v4(self):
    """Fail to connect with REST API failure (v4)"""
    self.mock_calls[self.action_format.format('addMappedSdc')] = self.MockHTTPSResponse(dict(errorCode=self.connector.VOLUME_NOT_MAPPED_ERROR_v4, message='Test error map volume'), 500)
    self.assertRaises(exception.BrickException, self.test_connect_volume)