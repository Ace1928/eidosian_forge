from __future__ import (absolute_import, division, print_function)
import json
import os
import base64
from urllib.error import HTTPError, URLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def check_license_id(self, license_id):
    """
        Check the license ID for a given operation.

        :param self: The object instance.
        :param module: The Ansible module.
        :param license_id: The ID of the license to check.
        :param operation: The operation to perform.
        :return: The response from the license URL.
        """
    license_uri = self.get_license_url()
    license_url = license_uri + f'/{license_id}'
    try:
        response = self.idrac.invoke_request(license_url, 'GET')
        return response
    except Exception:
        self.module.exit_json(msg=INVALID_LICENSE_MSG.format(license_id=license_id), skipped=True)