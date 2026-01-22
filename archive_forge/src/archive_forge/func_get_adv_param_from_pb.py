from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_adv_param_from_pb(self):
    """ Provide all the advance parameters named as required by SDK

        :return: all given advanced parameters
        :rtype: dict
        """
    param = {}
    LOG.info('Getting all given advance parameter')
    host_dict = self.get_host_dict_from_pb()
    if host_dict:
        param.update(host_dict)
    fields = ('description', 'anonymous_uid', 'anonymous_gid')
    for field in fields:
        if self.module.params[field] is not None:
            param[field] = self.module.params[field]
    if self.module.params['min_security'] and self.module.params['min_security'] in utils.NFSShareSecurityEnum.__members__:
        LOG.info('Getting min_security object from NFSShareSecurityEnum')
        param['min_security'] = utils.NFSShareSecurityEnum[self.module.params['min_security']]
    if self.module.params['default_access']:
        param['default_access'] = self.get_default_access()
    LOG.info('Successfully got advance parameter: %s', param)
    return param