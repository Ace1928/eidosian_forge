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
def check_delete_server_complete(self, server_id):
    """Wait for server to disappear from Nova."""
    try:
        server = self.fetch_server(server_id)
    except Exception as exc:
        self.ignore_not_found(exc)
        return True
    if not server:
        return False
    task_state_in_nova = getattr(server, 'OS-EXT-STS:task_state', None)
    if task_state_in_nova == 'deleting':
        return False
    status = self.get_status(server)
    if status == 'DELETED':
        return True
    if status == 'SOFT_DELETED':
        self.client().servers.force_delete(server_id)
    elif status == 'ERROR':
        fault = getattr(server, 'fault', {})
        message = fault.get('message', 'Unknown')
        code = fault.get('code')
        errmsg = _('Server %(name)s delete failed: (%(code)s) %(message)s') % dict(name=server.name, code=code, message=message)
        raise exception.ResourceInError(resource_status=status, status_reason=errmsg)
    return False