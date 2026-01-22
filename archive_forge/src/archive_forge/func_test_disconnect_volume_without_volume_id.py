import json
import os
from unittest import mock
import requests
from os_brick import exception
from os_brick.initiator.connectors import scaleio
from os_brick.tests.initiator import test_connector
def test_disconnect_volume_without_volume_id(self):
    """Successful disconnect from volume without a Volume Id"""
    connection_properties = dict(self.fake_connection_properties)
    connection_properties.pop('scaleIO_volume_id')
    self.connector.disconnect_volume(connection_properties, None)
    self.get_guid_mock.assert_called_once_with(self.connector.GET_GUID_OP_CODE)