from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def remove_volumes_from_cg(self, cg_name, volumes):
    """Remove volumes from consistency group.
            :param cg_name: The name of the consistency group
            :param volumes: The list of volumes to be removed
            :return: Boolean value to indicate if volumes are removed from
             consistency group
        """
    cg_details = self.unity_conn.get_cg(name=cg_name)._get_properties()
    existing_volumes_in_cg = cg_details['luns']
    existing_vol_ids = []
    if existing_volumes_in_cg:
        existing_vol_ids = [vol['UnityLun']['id'] for vol in existing_volumes_in_cg['UnityLunList']]
    ids_to_remove = []
    vol_name_list = []
    vol_id_list = []
    for vol in volumes:
        if 'vol_id' in vol and (not vol['vol_id'] in vol_id_list):
            vol_id_list.append(vol['vol_id'])
        elif 'vol_name' in vol and (not vol['vol_name'] in vol_name_list):
            vol_name_list.append(vol['vol_name'])
    'remove volume by name'
    for vol in vol_name_list:
        ids_to_remove.append(self.get_volume_details(vol_name=vol))
    vol_id_list = list(set(vol_id_list + ids_to_remove))
    ids_to_remove = list(set(existing_vol_ids).intersection(set(vol_id_list)))
    LOG.info('Volume IDs to remove %s', ids_to_remove)
    if len(ids_to_remove) == 0:
        return False
    vol_remove_list = []
    for vol in ids_to_remove:
        vol_dict = {'id': vol}
        vol_remove_list.append(vol_dict)
    cg_obj = self.return_cg_instance(cg_name)
    try:
        cg_obj.modify(lun_remove=vol_remove_list)
        return True
    except Exception as e:
        errormsg = 'Remove existing volumes from consistency group {0} failed with error {1}'.format(cg_name, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)