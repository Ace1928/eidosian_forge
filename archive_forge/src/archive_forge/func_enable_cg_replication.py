from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def enable_cg_replication(self, cg_name, replication):
    """ Add replication to the consistency group """
    try:
        self.validate_cg_replication_params(replication)
        cg_object = self.return_cg_instance(cg_name)
        if cg_object.check_cg_is_replicated():
            return False
        self.update_replication_params(replication)
        replication_args_list = {'dst_pool_id': replication['destination_pool_id']}
        if 'replication_mode' in replication and replication['replication_mode'] == 'asynchronous':
            replication_args_list['max_time_out_of_sync'] = replication['rpo']
        else:
            replication_args_list['max_time_out_of_sync'] = -1
        if 'replication_type' in replication and replication['replication_type'] == 'remote':
            remote_system_name = replication['remote_system_name']
            remote_system_list = self.unity_conn.get_remote_system()
            for remote_system in remote_system_list:
                if remote_system.name == remote_system_name:
                    replication_args_list['remote_system'] = remote_system
                    break
            if 'remote_system' not in replication_args_list.keys():
                errormsg = 'Remote system %s is not found' % remote_system_name
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
        source_lun_list = cg_object.luns
        replication_args_list['source_luns'] = self.get_destination_cg_luns(source_lun_list)
        if 'destination_cg_name' in replication and replication['destination_cg_name'] is not None:
            replication_args_list['dst_cg_name'] = replication['destination_cg_name']
        else:
            replication_args_list['dst_cg_name'] = 'DR_' + cg_object.name
        LOG.info(('Enabling replication to the consistency group %s', cg_object.name))
        cg_object.replicate_cg_with_dst_resource_provisioning(**replication_args_list)
        return True
    except Exception as e:
        errormsg = 'Enabling replication to the consistency group %s failed with error %s' % (cg_object.name, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)