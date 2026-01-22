from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_httpsWrite_credentials(self, CredentialDetails, global_credentials):
    """
        Get the current httpsWrite Credential from
        Cisco DNA Center based on the provided playbook details.
        Check this API using the check_return_status.

        Parameters:
            CredentialDetails (dict) - Playbook details containing Global Device Credentials.
            global_credentials (dict) - All global device credentials details.

        Returns:
            httpsWriteDetails (List) - The current httpsWrite.
        """
    all_httpsWrite = CredentialDetails.get('https_write')
    httpsWrite_details = global_credentials.get('httpsWrite')
    httpsWriteDetails = []
    if all_httpsWrite and httpsWrite_details:
        for httpsWriteCredential in all_httpsWrite:
            httpsWriteDetail = None
            httpsWriteId = httpsWriteCredential.get('id')
            if httpsWriteId:
                httpsWriteDetail = get_dict_result(httpsWrite_details, 'id', httpsWriteId)
                if not httpsWriteDetail:
                    self.msg = 'httpsWrite credential Id is invalid'
                    self.status = 'failed'
                    return self
            httpsWriteOldDescription = httpsWriteCredential.get('old_description')
            httpsWriteOldUsername = httpsWriteCredential.get('old_username')
            if httpsWriteOldDescription and httpsWriteOldUsername and (not httpsWriteDetail):
                for item in httpsWrite_details:
                    if item.get('description') == httpsWriteOldDescription and item.get('username') == httpsWriteOldUsername:
                        if httpsWriteDetail:
                            self.msg = 'More than one httpsWrite credential with same                                             old_description and old_username. Pass ID'
                            self.status = 'failed'
                            return self
                        httpsWriteDetail = item
                if not httpsWriteDetail:
                    self.msg = 'httpsWrite credential old_description or                                     old_username is invalid'
                    self.status = 'failed'
                    return self
            httpsWriteDescription = httpsWriteCredential.get('description')
            httpsWriteUsername = httpsWriteCredential.get('username')
            if httpsWriteDescription and httpsWriteUsername and (not httpsWriteDetail):
                for item in httpsWrite_details:
                    if item.get('description') == httpsWriteDescription and item.get('username') == httpsWriteUsername:
                        httpsWriteDetail = item
            httpsWriteDetails.append(httpsWriteDetail)
    return httpsWriteDetails