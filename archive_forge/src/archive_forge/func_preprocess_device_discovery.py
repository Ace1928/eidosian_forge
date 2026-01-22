from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
import time
import re
def preprocess_device_discovery(self, ip_address_list=None):
    """
        Preprocess the devices' information. Extract the IP addresses from
        the list of devices and perform additional processing based on the
        'discovery_type' in the validated configuration.

        Parameters:
          - ip_address_list: The list of devices' IP addresses intended for preprocessing.
                             If not provided, an empty list will be used.

        Returns:
          - ip_address_list: It returns IP address list for the API to process. The value passed
                             for single, CDP, LLDP, CIDR, Range and Multi Range varies depending
                             on the need.
        """
    if ip_address_list is None:
        ip_address_list = []
    discovery_type = self.validated_config[0].get('discovery_type')
    self.log('Discovery type passed for the discovery is {0}'.format(discovery_type), 'INFO')
    if discovery_type in ['SINGLE', 'CDP', 'LLDP']:
        if len(ip_address_list) == 1:
            ip_address_list = ip_address_list[0]
        else:
            self.preprocess_device_discovery_handle_error()
    elif discovery_type == 'CIDR':
        if len(ip_address_list) == 1:
            cidr_notation = ip_address_list[0]
            if len(cidr_notation.split('/')) == 2:
                ip_address_list = cidr_notation
            else:
                ip_address_list = '{0}/30'.format(cidr_notation)
                self.log('CIDR notation is being used for discovery and it requires a prefix length to be specified, such as 1.1.1.1/24.                        As no prefix length was provided, it will default to 30.', 'WARNING')
        else:
            self.preprocess_device_discovery_handle_error()
    elif discovery_type == 'RANGE':
        if len(ip_address_list) == 1:
            if len(str(ip_address_list[0]).split('-')) == 2:
                ip_address_list = ip_address_list[0]
            else:
                ip_address_list = '{0}-{1}'.format(ip_address_list[0], ip_address_list[0])
        else:
            self.preprocess_device_discovery_handle_error()
    else:
        new_ip_collected = []
        for ip in ip_address_list:
            if len(str(ip).split('-')) != 2:
                ip_collected = '{0}-{0}'.format(ip)
                new_ip_collected.append(ip_collected)
            else:
                new_ip_collected.append(ip)
        ip_address_list = ','.join(new_ip_collected)
    self.log('Collected IP address/addresses are {0}'.format(str(ip_address_list)), 'INFO')
    return str(ip_address_list)