from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_host_obj(self, host_id=None, host_name=None, ip_address=None):
    """
        Get host object
        :param host_id: ID of the host
        :param host_name: Name of the host
        :param ip_address: Network address of the host
        :return: Host object
        :rtype: object
        """
    try:
        host_obj = None
        host = None
        if host_id:
            host = host_id
            host_obj = self.unity.get_host(_id=host_id)
        elif host_name:
            host = host_name
            host_obj = self.unity.get_host(name=host_name)
        elif ip_address:
            host = ip_address
            host_obj = self.unity.get_host(address=ip_address)
        if host_obj and host_obj.existed:
            LOG.info('Successfully got host: %s', host_obj.name)
            return host_obj
        else:
            msg = f'Host : {host} does not exists'
            LOG.error(msg)
            self.module.fail_json(msg=msg)
    except Exception as e:
        msg = f'Failed to get host {host}, error: {e}'
        LOG.error(msg)
        self.module.fail_json(msg=msg)