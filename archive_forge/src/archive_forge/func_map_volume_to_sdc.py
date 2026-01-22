from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
import copy
def map_volume_to_sdc(self, volume, sdc):
    """Map SDC's to volume
            :param volume: volume details
            :param sdc: List of SDCs
            :return: Boolean indicating if mapping operation is successful
        """
    current_sdcs = volume['mappedSdcInfo']
    current_sdc_ids = []
    sdc_id_list = []
    sdc_map_list = []
    sdc_modify_list1 = []
    sdc_modify_list2 = []
    if current_sdcs:
        for temp in current_sdcs:
            current_sdc_ids.append(temp['sdcId'])
    for temp in sdc:
        if 'sdc_name' in temp and temp['sdc_name']:
            sdc_id = self.get_sdc_id(sdc_name=temp['sdc_name'])
        elif 'sdc_ip' in temp and temp['sdc_ip']:
            sdc_id = self.get_sdc_id(sdc_ip=temp['sdc_ip'])
        else:
            sdc_id = self.get_sdc_id(sdc_id=temp['sdc_id'])
        if sdc_id not in current_sdc_ids:
            sdc_id_list.append(sdc_id)
            temp['sdc_id'] = sdc_id
            if 'access_mode' in temp:
                temp['access_mode'] = get_access_mode(temp['access_mode'])
            if 'bandwidth_limit' not in temp:
                temp['bandwidth_limit'] = None
            if 'iops_limit' not in temp:
                temp['iops_limit'] = None
            sdc_map_list.append(temp)
        else:
            access_mode_dict, limits_dict = check_for_sdc_modification(volume, sdc_id, temp)
            if access_mode_dict:
                sdc_modify_list1.append(access_mode_dict)
            if limits_dict:
                sdc_modify_list2.append(limits_dict)
    LOG.info('SDC to add: %s', sdc_map_list)
    if not sdc_map_list:
        return (False, sdc_modify_list1, sdc_modify_list2)
    try:
        changed = False
        for sdc in sdc_map_list:
            payload = {'volume_id': volume['id'], 'sdc_id': sdc['sdc_id'], 'access_mode': sdc['access_mode'], 'allow_multiple_mappings': self.module.params['allow_multiple_mappings']}
            self.powerflex_conn.volume.add_mapped_sdc(**payload)
            if sdc['bandwidth_limit'] or sdc['iops_limit']:
                payload = {'volume_id': volume['id'], 'sdc_id': sdc['sdc_id'], 'bandwidth_limit': sdc['bandwidth_limit'], 'iops_limit': sdc['iops_limit']}
                self.powerflex_conn.volume.set_mapped_sdc_limits(**payload)
            changed = True
        return (changed, sdc_modify_list1, sdc_modify_list2)
    except Exception as e:
        errormsg = 'Mapping volume {0} to SDC {1} failed with error {2}'.format(volume['name'], sdc['sdc_id'], str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)