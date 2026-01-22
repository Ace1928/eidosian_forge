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
def associate_floatingip(self, server_id, floatingip_id):
    iface_list = self.fetch_server(server_id).interface_list()
    if len(iface_list) == 0:
        raise client_exception.InterfaceNotFound(id=server_id)
    if len(iface_list) > 1:
        LOG.warning('Multiple interfaces found for server %s, using the first one.', server_id)
    port_id = iface_list[0].port_id
    fixed_ips = iface_list[0].fixed_ips
    fixed_address = next((ip['ip_address'] for ip in fixed_ips if netutils.is_valid_ipv4(ip['ip_address'])))
    request_body = {'floatingip': {'port_id': port_id, 'fixed_ip_address': fixed_address}}
    self.clients.client('neutron').update_floatingip(floatingip_id, request_body)