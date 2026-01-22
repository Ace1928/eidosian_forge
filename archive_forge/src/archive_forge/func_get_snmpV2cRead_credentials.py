from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_snmpV2cRead_credentials(self, CredentialDetails, global_credentials):
    """
        Get the current snmpV2cRead Credential from
        Cisco DNA Center based on the provided playbook details.
        Check this API using the check_return_status.

        Parameters:
            CredentialDetails (dict) - Playbook details containing Global Device Credentials.
            global_credentials (dict) - All global device credentials details.

        Returns:
            snmpV2cReadDetails (List) - The current snmpV2cRead.
        """
    all_snmpV2cRead = CredentialDetails.get('snmp_v2c_read')
    snmpV2cRead_details = global_credentials.get('snmpV2cRead')
    snmpV2cReadDetails = []
    if all_snmpV2cRead and snmpV2cRead_details:
        for snmpV2cReadCredential in all_snmpV2cRead:
            snmpV2cReadDetail = None
            snmpV2cReadId = snmpV2cReadCredential.get('id')
            if snmpV2cReadId:
                snmpV2cReadDetail = get_dict_result(snmpV2cRead_details, 'id', snmpV2cReadId)
                if not snmpV2cReadDetail:
                    self.msg = 'snmpV2cRead credential ID is invalid'
                    self.status = 'failed'
                    return self
            snmpV2cReadOldDescription = snmpV2cReadCredential.get('old_description')
            if snmpV2cReadOldDescription and (not snmpV2cReadDetail):
                snmpV2cReadDetail = get_dict_result(snmpV2cRead_details, 'description', snmpV2cReadOldDescription)
                if not snmpV2cReadDetail:
                    self.msg = 'snmpV2cRead credential old_description is invalid'
                    self.status = 'failed'
                    return self
            snmpV2cReadDescription = snmpV2cReadCredential.get('description')
            if snmpV2cReadDescription and (not snmpV2cReadDetail):
                snmpV2cReadDetail = get_dict_result(snmpV2cRead_details, 'description', snmpV2cReadDescription)
            snmpV2cReadDetails.append(snmpV2cReadDetail)
    return snmpV2cReadDetails