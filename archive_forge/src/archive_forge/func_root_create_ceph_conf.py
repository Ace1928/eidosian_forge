from typing import Optional
from oslo_utils import fileutils
from oslo_utils import importutils
import os_brick.privileged
@os_brick.privileged.default.entrypoint
def root_create_ceph_conf(monitor_ips, monitor_ports, cluster_name, user, keyring):
    """Create a .conf file for Ceph cluster only accessible by root."""
    get_rbd_class()
    assert RBDConnector is not None
    return RBDConnector._create_ceph_conf(monitor_ips, monitor_ports, cluster_name, user, keyring)