import uuid
import base64
from openstackclient.identity import common as identity_common
import os
from oslo_utils import encodeutils
from oslo_utils import uuidutils
import prettytable
import simplejson as json
import sys
from troveclient.apiclient import exceptions
def decode_data(data):
    """Encode the data using the base64 codec."""
    return bytearray([item for item in base64.b64decode(data)])