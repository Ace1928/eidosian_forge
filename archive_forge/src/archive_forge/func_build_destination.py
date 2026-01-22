from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_obj
from ansible.module_utils._text import to_native
@staticmethod
def build_destination(dest_hostname, dest_port, dest_community):
    """Build destination spec"""
    destination = vim.host.SnmpSystem.SnmpConfigSpec.Destination()
    destination.hostName = dest_hostname
    destination.port = dest_port
    destination.community = dest_community
    return destination