from __future__ import annotations
import os
import tempfile
import typing
from typing import Any, Optional, Union  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import excutils
from oslo_utils import fileutils
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import base_rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rbd as rbd_privsep
from os_brick import utils
@classmethod
def create_non_openstack_config(cls, connection_properties: dict[str, Any]):
    """Get root owned Ceph's .conf file for non OpenStack usage."""
    keyring = connection_properties.get('keyring')
    if not keyring:
        return None
    try:
        user = connection_properties['auth_username']
        pool, volume = connection_properties['name'].split('/')
        cluster_name = connection_properties['cluster_name']
        monitor_ips = connection_properties['hosts']
        monitor_ports = connection_properties['ports']
        keyring = connection_properties.get('keyring')
    except (KeyError, ValueError):
        msg = _('Connect volume failed, malformed connection properties.')
        raise exception.BrickException(msg=msg)
    conf = rbd_privsep.root_create_ceph_conf(monitor_ips, monitor_ports, str(cluster_name), user, keyring)
    return conf