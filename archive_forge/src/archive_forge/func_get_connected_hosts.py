import logging
import posixpath
import random
import re
import http.client as httplib
import urllib.parse as urlparse
from oslo_vmware._i18n import _
from oslo_vmware import constants
from oslo_vmware import exceptions
from oslo_vmware import vim_util
def get_connected_hosts(self, session):
    """Get a list of usable (accessible, mounted, read-writable) hosts where
        the datastore is mounted.

        :param: session: session
        :return: list of HostSystem managed object references
        """
    hosts = []
    summary = self.get_summary(session)
    if not summary.accessible:
        return hosts
    host_mounts = session.invoke_api(vim_util, 'get_object_property', session.vim, self.ref, 'host')
    if not hasattr(host_mounts, 'DatastoreHostMount'):
        return hosts
    for host_mount in host_mounts.DatastoreHostMount:
        if self.is_datastore_mount_usable(host_mount.mountInfo):
            hosts.append(host_mount.key)
    connectables = []
    if hosts:
        host_runtimes = session.invoke_api(vim_util, 'get_properties_for_a_collection_of_objects', session.vim, 'HostSystem', hosts, ['runtime'])
        for host_object in host_runtimes.objects:
            host_props = vim_util.propset_dict(host_object.propSet)
            host_runtime = host_props.get('runtime')
            if hasattr(host_runtime, 'inMaintenanceMode') and (not host_runtime.inMaintenanceMode):
                connectables.append(host_object.obj)
    return connectables