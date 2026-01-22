import collections
import email
from email.mime import multipart
from email.mime import text
import os
import pkgutil
import string
from urllib import parse as urlparse
from neutronclient.common import exceptions as q_exceptions
from novaclient import api_versions
from novaclient import client as nc
from novaclient import exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import netutils
import tenacity
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_exception
from heat.engine.clients import client_plugin
from heat.engine.clients import microversion_mixin
from heat.engine.clients import os as os_client
from heat.engine import constraints
@staticmethod
def build_ignition_data(metadata, userdata):
    if not metadata:
        return userdata
    payload = jsonutils.loads(userdata)
    encoded_metadata = urlparse.quote(jsonutils.dumps(metadata))
    path_list = ['/var/lib/heat-cfntools/cfn-init-data', '/var/lib/cloud/data/cfn-init-data']
    ignition_format_metadata = {'filesystem': 'root', 'group': {'name': 'root'}, 'path': '', 'user': {'name': 'root'}, 'contents': {'source': 'data:,' + encoded_metadata, 'verification': {}}, 'mode': 416}
    for path in path_list:
        storage = payload.setdefault('storage', {})
        try:
            files = storage.setdefault('files', [])
        except AttributeError:
            raise ValueError('Ignition "storage" section must be a map')
        else:
            try:
                data = ignition_format_metadata.copy()
                data['path'] = path
                files.append(data)
            except AttributeError:
                raise ValueError('Ignition "files" section must be a list')
    return jsonutils.dumps(payload)