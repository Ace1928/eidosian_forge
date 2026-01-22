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
def check_verify_resize(self, server_id):
    server = self.fetch_server(server_id)
    if not server:
        return False
    status = self.get_status(server)
    if status == 'ACTIVE':
        return True
    if status == 'VERIFY_RESIZE':
        return False
    task_state_in_nova = getattr(server, 'OS-EXT-STS:task_state', None)
    if task_state_in_nova is not None and 'resize' in task_state_in_nova:
        return False
    else:
        msg = _('Confirm resize for server %s failed') % server_id
        raise exception.ResourceUnknownStatus(result=msg, resource_status=status)