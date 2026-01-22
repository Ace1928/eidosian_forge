from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_snmpV3_credentials(self, CredentialDetails, global_credentials):
    """
        Get the current snmpV3 Credential from
        Cisco DNA Center based on the provided playbook details.
        Check this API using the check_return_status.

        Parameters:
            CredentialDetails (dict) - Playbook details containing Global Device Credentials.
            global_credentials (dict) - All global device credentials details.

        Returns:
            snmpV3Details (List) - The current snmpV3.
        """
    all_snmpV3 = CredentialDetails.get('snmp_v3')
    snmpV3_details = global_credentials.get('snmpV3')
    snmpV3Details = []
    if all_snmpV3 and snmpV3_details:
        for snmpV3Credential in all_snmpV3:
            snmpV3Detail = None
            snmpV3Id = snmpV3Credential.get('id')
            if snmpV3Id:
                snmpV3Detail = get_dict_result(snmpV3_details, 'id', snmpV3Id)
                if not snmpV3Detail:
                    self.msg = 'snmpV3 credential id is invalid'
                    self.status = 'failed'
                    return self
            snmpV3OldDescription = snmpV3Credential.get('old_description')
            if snmpV3OldDescription and (not snmpV3Detail):
                snmpV3Detail = get_dict_result(snmpV3_details, 'description', snmpV3OldDescription)
                if not snmpV3Detail:
                    self.msg = 'snmpV3 credential old_description is invalid'
                    self.status = 'failed'
                    return self
            snmpV3Description = snmpV3Credential.get('description')
            if snmpV3Description and (not snmpV3Detail):
                snmpV3Detail = get_dict_result(snmpV3_details, 'description', snmpV3Description)
            snmpV3Details.append(snmpV3Detail)
    return snmpV3Details