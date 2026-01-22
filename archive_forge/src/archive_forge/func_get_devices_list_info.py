from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
import time
import re
def get_devices_list_info(self):
    """
        Retrieve the list of devices from the validated configuration.
        It then updates the result attribute with this list.

        Returns:
          - ip_address_list: The list of devices extracted from the
                          'validated_config' attribute.
        """
    ip_address_list = self.validated_config[0].get('ip_address_list')
    self.result.update(dict(devices_info=ip_address_list))
    self.log('Details of the device list passed: {0}'.format(str(ip_address_list)), 'INFO')
    return ip_address_list