from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_snmpV2cWrite_params(self, snmpV2cWriteDetails):
    """
        Format the snmpV2cWrite parameters for the snmpV2cWrite
        credential configuration in Cisco DNA Center.

        Parameters:
            snmpV2cWriteDetails (list of dict) - Cisco DNA Center
            Details containing snmpV2cWrite Credentials.

        Returns:
            snmpV2cWrite (list of dict) - Processed snmpV2cWrite credential
            data in the format suitable for the Cisco DNA Center config.
        """
    snmpV2cWrite = []
    for item in snmpV2cWriteDetails:
        if item is None:
            snmpV2cWrite.append(None)
        else:
            value = {'description': item.get('description'), 'id': item.get('id')}
            snmpV2cWrite.append(value)
    return snmpV2cWrite