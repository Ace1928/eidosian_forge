from __future__ import (absolute_import, division, print_function)
import json
import re
import time
import os
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.parameters import env_fallback
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
def get_idrac_local_account_attr(self, idrac_attribues, fqdd=None):
    """
        This method filtered from all the user attributes from the given idrac attributes.
        :param idrac_attribues: all the idrac attribues in json data format.
        :return: user attributes in dictionary format
        """
    user_attr = None
    if 'SystemConfiguration' in idrac_attribues:
        sys_config = idrac_attribues.get('SystemConfiguration')
        for comp in sys_config.get('Components'):
            if comp.get('FQDD') == fqdd:
                attributes = comp.get('Attributes')
                break
        user_attr = dict([(attr['Name'], attr['Value']) for attr in attributes if attr['Name'].startswith('Users.')])
    return user_attr