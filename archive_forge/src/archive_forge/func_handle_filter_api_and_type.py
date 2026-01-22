from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils import arguments, errors, utils
def handle_filter_api_and_type(payload_filters, workflow):
    filter_count = 0
    for filter in workflow['filters']:
        payload_filters[filter_count]['type'] = FILTER_TYPE[filter['type']]
        payload_filters[filter_count]['api_version'] = API_VERSION['v2']
        filter_count += 1