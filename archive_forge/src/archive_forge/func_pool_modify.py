from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def pool_modify(self, id, new_pool_name, pool_description, fast_cache, fast_vp):
    """ Modify attributes of storage pool """
    pool_obj = utils.UnityPool.get(self.conn._cli, id)
    try:
        pool_obj.modify(name=new_pool_name, description=pool_description, is_fast_cache_enabled=fast_cache, is_fastvp_enabled=fast_vp)
        new_storage_pool_details = self.get_details(pool_id=id, pool_name=None)
        LOG.info('Modification Successful')
        return new_storage_pool_details
    except Exception as e:
        if self.module.params['pool_id']:
            pool_identifier = self.module.params['pool_id']
        else:
            pool_identifier = self.module.params['pool_name']
        error_message = 'Modify attributes of storage pool {0} failed with error: {1}'.format(pool_identifier, str(e))
        LOG.error(error_message)
        self.module.fail_json(msg=error_message)