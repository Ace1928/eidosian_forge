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
def check_rebuild(self, server_id):
    """Verify that a rebuilding server is rebuilt.

        Raise error if it ends up in an ERROR state.
        """
    server = self.fetch_server(server_id)
    if server is None or server.status == 'REBUILD':
        return False
    if server.status == 'ERROR':
        raise exception.Error(_("Rebuilding server failed, status '%s'") % server.status)
    else:
        return True