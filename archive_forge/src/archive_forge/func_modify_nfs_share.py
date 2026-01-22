from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def modify_nfs_share(self, nfs_obj):
    """ Modify given nfs share

        :param nfs_obj: NFS share obj
        :type nfs_obj: UnityNfsShare
        :return: tuple(bool, nfs_obj)
            - bool: indicates whether nfs_obj is modified or not
            - nfs_obj: same nfs_obj if not modified else modified nfs_obj
        :rtype: tuple
        """
    modify_param = {}
    LOG.info('Modifying nfs share')
    nfs_details = nfs_obj._get_properties()
    fields = ('description', 'anonymous_uid', 'anonymous_gid')
    for field in fields:
        if self.module.params[field] is not None and self.module.params[field] != nfs_details[field]:
            modify_param[field] = self.module.params[field]
    if self.module.params['min_security'] and self.module.params['min_security'] != nfs_obj.min_security.name:
        modify_param['min_security'] = utils.NFSShareSecurityEnum[self.module.params['min_security']]
    if self.module.params['default_access']:
        default_access = self.get_default_access()
        if default_access != nfs_obj.default_access:
            modify_param['default_access'] = default_access
    new_host_dict = self.get_host_dict_from_pb()
    if new_host_dict:
        try:
            if is_nfs_have_host_with_host_obj(nfs_details) and (not self.module.params['adv_host_mgmt_enabled']):
                msg = 'Modification of nfs host is restricted using adv_host_mgmt_enabled as false since nfs already have host added using host obj'
                LOG.error(msg)
                self.module.fail_json(msg=msg)
            elif is_nfs_have_host_with_host_string(nfs_details) and self.module.params['adv_host_mgmt_enabled']:
                msg = 'Modification of nfs host is restricted using adv_host_mgmt_enabled as true since nfs already have host added without host obj'
                LOG.error(msg)
                self.module.fail_json(msg=msg)
            LOG.info('Extracting same given param from nfs')
            existing_host_dict = {k: nfs_details[k] for k in new_host_dict}
        except KeyError as e:
            msg = 'Failed to extract key-value from current nfs: %s' % str(e)
            LOG.error(msg)
            self.module.fail_json(msg=msg)
        if self.module.params['host_state'] == HOST_STATE_LIST[0]:
            LOG.info('Getting host to be added')
            modify_host_dict = self.add_host(existing_host_dict, new_host_dict)
        else:
            LOG.info('Getting host to be removed')
            modify_host_dict = self.remove_host(existing_host_dict, new_host_dict)
        if modify_host_dict:
            modify_param.update(modify_host_dict)
    if not modify_param:
        LOG.info('Existing nfs attribute value is same as given input, so returning same nfs object - idempotency case')
        return (False, nfs_obj)
    modify_param = self.correct_payload_as_per_sdk(modify_param, nfs_details)
    try:
        resp = nfs_obj.modify(**modify_param)
        resp.raise_if_err()
    except Exception as e:
        msg = 'Failed to modify nfs error: %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)
    return (True, self.get_nfs_share(id=nfs_obj.id))