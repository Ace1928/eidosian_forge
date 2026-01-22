from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def reconnect_host(self, host_object):
    """Reconnect host to vCenter"""
    reconnecthost_args = {}
    reconnecthost_args['reconnectSpec'] = vim.HostSystem.ReconnectSpec()
    reconnecthost_args['reconnectSpec'].syncState = True
    if self.esxi_username and self.esxi_password:
        reconnecthost_args['cnxSpec'] = self.get_host_connect_spec()
    try:
        task = host_object.ReconnectHost_Task(**reconnecthost_args)
    except vim.fault.InvalidLogin as invalid_login:
        self.module.fail_json(msg='Cannot authenticate with the host : %s' % to_native(invalid_login))
    except vim.fault.InvalidState as invalid_state:
        self.module.fail_json(msg='The host is not disconnected : %s' % to_native(invalid_state))
    except vim.fault.InvalidName as invalid_name:
        self.module.fail_json(msg='The host name is invalid : %s' % to_native(invalid_name))
    except vim.fault.HostConnectFault as connect_fault:
        self.module.fail_json(msg='An error occurred during reconnect : %s' % to_native(connect_fault))
    except vmodl.fault.NotSupported as not_supported:
        self.module.fail_json(msg='No host can be added to this group : %s' % to_native(not_supported))
    except vim.fault.AlreadyBeingManaged as already_managed:
        self.module.fail_json(msg='The host is already being managed by another vCenter server : %s' % to_native(already_managed))
    except vmodl.fault.NotEnoughLicenses as not_enough_licenses:
        self.module.fail_json(msg='There are not enough licenses to add this host : %s' % to_native(not_enough_licenses))
    except vim.fault.NoHost as no_host:
        self.module.fail_json(msg='Unable to contact the host : %s' % to_native(no_host))
    except vim.fault.NotSupportedHost as host_not_supported:
        self.module.fail_json(msg='The host is running a software version that is not supported : %s' % to_native(host_not_supported))
    except vim.fault.SSLVerifyFault as ssl_fault:
        self.module.fail_json(msg='The host certificate could not be authenticated : %s' % to_native(ssl_fault))
    try:
        changed, result = wait_for_task(task)
    except TaskError as task_error:
        self.module.fail_json(msg="Failed to reconnect host to vCenter '%s' due to %s" % (self.vcenter, to_native(task_error)))