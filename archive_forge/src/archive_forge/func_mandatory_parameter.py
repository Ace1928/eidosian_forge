from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def mandatory_parameter(self):
    """
        Check for and validate mandatory parameters for adding network devices in Cisco Catalyst Center.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Cisco Catalyst Center.
        Returns:
            dict: The input `config` dictionary if all mandatory parameters are present.
        Description:
            It will check the mandatory parameters for adding the devices in Cisco Catalyst Center.
        """
    device_type = self.config[0].get('type', 'NETWORK_DEVICE')
    params_dict = {'NETWORK_DEVICE': ['ip_address_list', 'password', 'username'], 'COMPUTE_DEVICE': ['ip_address_list', 'http_username', 'http_password', 'http_port'], 'MERAKI_DASHBOARD': ['http_password'], 'FIREPOWER_MANAGEMENT_SYSTEM': ['ip_address_list', 'http_username', 'http_password'], 'THIRD_PARTY_DEVICE': ['ip_address_list']}
    params_list = params_dict.get(device_type, [])
    mandatory_params_absent = []
    for param in params_list:
        if param not in self.config[0]:
            mandatory_params_absent.append(param)
    if mandatory_params_absent:
        self.status = 'failed'
        self.msg = 'Required parameters {0} for adding devices are not present'.format(str(mandatory_params_absent))
        self.result['msg'] = self.msg
        self.log(self.msg, 'ERROR')
    else:
        self.status = 'success'
        self.msg = 'Required parameter for Adding the devices in Inventory are present.'
        self.log(self.msg, 'INFO')
    return self