from __future__ import (absolute_import, division, print_function)
import os
import json
import time
from ssl import SSLError
from xml.etree import ElementTree as ET
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_jobid(module, resp):
    """Get the Job ID from the response header."""
    jobid = None
    if resp.status_code == 202:
        joburi = resp.headers.get('Location')
        if joburi is None:
            module.fail_json(msg='Failed to update firmware.')
        jobid = joburi.split('/')[-1]
    else:
        module.fail_json(msg='Failed to update firmware.')
    return jobid