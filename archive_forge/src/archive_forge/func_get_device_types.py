from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_device_types(self, device_types):
    """
        Store device types parameters from the playbook for template processing in DNAC.
        Check using check_return_status()

        Parameters:
            device_types (dict) - Device types details containing Template information.

        Returns:
            deviceTypes (dict) - Organized device types parameters.
        """
    if device_types is None:
        return None
    deviceTypes = []
    i = 0
    for item in device_types:
        deviceTypes.append({})
        product_family = item.get('product_family')
        if product_family is not None:
            deviceTypes[i].update({'productFamily': product_family})
        else:
            self.msg = 'product_family is mandatory for deviceTypes'
            self.status = 'failed'
            return self.check_return_status()
        product_series = item.get('product_series')
        if product_series is not None:
            deviceTypes[i].update({'productSeries': product_series})
        product_type = item.get('product_type')
        if product_type is not None:
            deviceTypes[i].update({'productType': product_type})
        i = i + 1
    return deviceTypes