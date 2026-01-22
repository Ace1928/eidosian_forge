from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def state_update_host(self):
    """Move host to a cluster or a folder, or vice versa"""
    changed = True
    result = None
    reconnect = False
    if self.reconnect_disconnected and self.host_update.runtime.connectionState == 'disconnected':
        reconnect = True
    parent_type = self.get_parent_type(self.host_update)
    if self.folder_name:
        if self.module.check_mode:
            if reconnect or self.state == 'add_or_reconnect' or self.state == 'reconnect':
                result = "Host would be reconnected and moved to folder '%s'" % self.folder_name
            else:
                result = "Host would be moved to folder '%s'" % self.folder_name
        else:
            if reconnect or self.state == 'add_or_reconnect' or self.state == 'reconnect':
                self.reconnect_host(self.host_update)
            try:
                try:
                    if parent_type == 'folder':
                        task = self.folder.MoveIntoFolder_Task([self.host_update.parent])
                    elif parent_type == 'cluster':
                        self.put_host_in_maintenance_mode(self.host_update)
                        task = self.folder.MoveIntoFolder_Task([self.host_update])
                except vim.fault.DuplicateName as duplicate_name:
                    self.module.fail_json(msg='The folder already contains an object with the specified name : %s' % to_native(duplicate_name))
                except vim.fault.InvalidFolder as invalid_folder:
                    self.module.fail_json(msg='The parent of this folder is in the list of objects : %s' % to_native(invalid_folder))
                except vim.fault.InvalidState as invalid_state:
                    self.module.fail_json(msg='Failed to move host, this can be due to either of following : 1. The host is not part of the same datacenter, 2. The host is not in maintenance mode : %s' % to_native(invalid_state))
                except vmodl.fault.NotSupported as not_supported:
                    self.module.fail_json(msg='The target folder is not a host folder : %s' % to_native(not_supported))
                except vim.fault.DisallowedOperationOnFailoverHost as failover_host:
                    self.module.fail_json(msg='The host is configured as a failover host : %s' % to_native(failover_host))
                except vim.fault.VmAlreadyExistsInDatacenter as already_exists:
                    self.module.fail_json(msg="The host's virtual machines are already registered to a host in the destination datacenter : %s" % to_native(already_exists))
                changed, result = wait_for_task(task)
            except TaskError as task_error_exception:
                task_error = task_error_exception.args[0]
                self.module.fail_json(msg='Failed to move host %s to folder %s due to %s' % (self.esxi_hostname, self.folder_name, to_native(task_error)))
            if reconnect or self.state == 'add_or_reconnect' or self.state == 'reconnect':
                result = "Host reconnected and moved to folder '%s'" % self.folder_name
            else:
                result = "Host moved to folder '%s'" % self.folder_name
    elif self.cluster_name:
        if self.module.check_mode:
            result = "Host would be moved to cluster '%s'" % self.cluster_name
        else:
            if parent_type == 'cluster':
                self.put_host_in_maintenance_mode(self.host_update)
            resource_pool = None
            try:
                try:
                    task = self.cluster.MoveHostInto_Task(host=self.host_update, resourcePool=resource_pool)
                except vim.fault.TooManyHosts as too_many_hosts:
                    self.module.fail_json(msg='No additional hosts can be added to the cluster : %s' % to_native(too_many_hosts))
                except vim.fault.InvalidState as invalid_state:
                    self.module.fail_json(msg='The host is already part of a cluster and is not in maintenance mode : %s' % to_native(invalid_state))
                except vmodl.fault.InvalidArgument as invalid_argument:
                    self.module.fail_json(msg='Failed to move host, this can be due to either of following : 1. The host is is not a part of the same datacenter as the cluster, 2. The source and destination clusters are the same : %s' % to_native(invalid_argument))
                changed, result = wait_for_task(task)
            except TaskError as task_error_exception:
                task_error = task_error_exception.args[0]
                self.module.fail_json(msg="Failed to move host to cluster '%s' due to : %s" % (self.cluster_name, to_native(task_error)))
            if reconnect or self.state == 'add_or_reconnect' or self.state == 'reconnect':
                result = "Host reconnected and moved to cluster '%s'" % self.cluster_name
            else:
                result = "Host moved to cluster '%s'" % self.cluster_name
    self.module.exit_json(changed=changed, msg=str(result))