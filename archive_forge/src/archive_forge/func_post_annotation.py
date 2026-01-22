from __future__ import absolute_import, division, print_function
import json
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six import PY3
from ansible.module_utils.common.text.converters import to_native
def post_annotation(annotation, api_key):
    """ Takes annotation dict and api_key string"""
    base_url = 'https://api.circonus.com/v2'
    anootate_post_endpoint = '/annotation'
    resp = requests.post(base_url + anootate_post_endpoint, headers=build_headers(api_key), data=json.dumps(annotation))
    resp.raise_for_status()
    return resp