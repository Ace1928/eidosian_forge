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
def get_resource_id_by_name(manager, name):
    resource = manager.find(name=name)
    return resource.id