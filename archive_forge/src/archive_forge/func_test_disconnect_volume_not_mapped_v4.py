import json
import os
from unittest import mock
import requests
from os_brick import exception
from os_brick.initiator.connectors import scaleio
from os_brick.tests.initiator import test_connector
def test_disconnect_volume_not_mapped_v4(self):
    """Ignore REST API failure for volume not mapped (v4)"""
    self.mock_calls[self.action_format.format('removeMappedSdc')] = self.MockHTTPSResponse(dict(errorCode=self.connector.VOLUME_NOT_MAPPED_ERROR_v4, message='Test error map volume'), 500)
    self.test_disconnect_volume()