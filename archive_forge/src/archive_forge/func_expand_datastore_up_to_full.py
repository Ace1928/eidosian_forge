from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, find_datastore_by_name, find_obj, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vmware_sms import SMS
from ansible.module_utils._text import to_native
def expand_datastore_up_to_full(self):
    """
        Expand a datastore capacity up to full if there is free capacity.
        """
    cnf_mng = self.esxi.configManager
    for datastore_obj in self.esxi.datastore:
        if datastore_obj.name == self.datastore_name:
            expand_datastore_obj = datastore_obj
            break
    vmfs_ds_options = cnf_mng.datastoreSystem.QueryVmfsDatastoreExpandOptions(expand_datastore_obj)
    if vmfs_ds_options:
        if self.module.check_mode is False:
            try:
                cnf_mng.datastoreSystem.ExpandVmfsDatastore(datastore=expand_datastore_obj, spec=vmfs_ds_options[0].spec)
            except Exception as e:
                self.module.fail_json(msg='%s can not expand the datastore: %s' % (to_native(e.msg), self.datastore_name))
        self.module.exit_json(changed=True)