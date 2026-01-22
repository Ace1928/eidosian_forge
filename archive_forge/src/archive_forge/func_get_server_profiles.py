from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import \
def get_server_profiles(module, rest_obj, service_tags):
    profile_dict = {}
    for stag in service_tags:
        prof = _get_profile(module, rest_obj, stag)
        intrfc = _get_interface(module, rest_obj, stag)
        prof['ServerInterfaceProfiles'] = intrfc
        prof = strip_substr_dict(prof)
        profile_dict[stag] = prof
    return profile_dict