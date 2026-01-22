from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_wireless_param(self, prov_dict):
    """
        Get wireless provisioning parameters for a device.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            prov_dict (dict): A dictionary containing configuration parameters for wireless provisioning.
        Returns:
            wireless_param (list of dict): A list containing a dictionary with wireless provisioning parameters.
        Description:
            This function constructs a list containing a dictionary with wireless provisioning parameters based on the
            configuration provided in the playbook. It validates the managed AP locations, ensuring they are of type "floor."
            The function then queries Cisco Catalyst Center to get network device details using the provided device IP.
            If the device is not found, the function returns the class instance with appropriate status and log messages and
            returns the wireless provisioning parameters containing site information, managed AP
            locations, dynamic interfaces, and device name.
        """
    try:
        device_ip_address = prov_dict['device_ip']
        site_name = prov_dict['site_name']
        wireless_param = [{'site': site_name, 'managedAPLocations': prov_dict['managed_ap_locations']}]
        for ap_loc in wireless_param[0]['managedAPLocations']:
            if self.get_site_type(site_name=ap_loc) != 'floor':
                self.status = 'failed'
                self.msg = 'Managed AP Location must be a floor'
                self.log(self.msg, 'ERROR')
                return self
        wireless_param[0]['dynamicInterfaces'] = []
        for interface in prov_dict.get('dynamic_interfaces'):
            interface_dict = {'interfaceIPAddress': interface.get('interface_ip_address'), 'interfaceNetmaskInCIDR': interface.get('interface_netmask_in_cidr'), 'interfaceGateway': interface.get('interface_gateway'), 'lagOrPortNumber': interface.get('lag_or_port_number'), 'vlanId': interface.get('vlan_id'), 'interfaceName': interface.get('interface_name')}
            wireless_param[0]['dynamicInterfaces'].append(interface_dict)
        response = self.dnac_apply['exec'](family='devices', function='get_network_device_by_ip', params={'ip_address': device_ip_address})
        response = response.get('response')
        wireless_param[0]['deviceName'] = response.get('hostname')
        self.wireless_param = wireless_param
        self.status = 'success'
        self.log('Successfully collected all the parameters required for Wireless Provisioning', 'DEBUG')
    except Exception as e:
        self.msg = "An exception occured while fetching the details for wireless provisioning of\n                device '{0}' due to - {1}".format(device_ip_address, str(e))
        self.log(self.msg, 'ERROR')
    return self