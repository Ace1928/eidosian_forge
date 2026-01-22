from __future__ import absolute_import, division, print_function
import random
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import fetch_url
def query_filter_list_by_name(self, path, key_name, result_key, param_key=None, key_id=None, query_params=None, get_details=False, fail_not_found=False, skip_transform=True):
    param_value = self.module.params.get(param_key or key_name)
    found = dict()
    for resource in self.query_list(path=path, result_key=result_key, query_params=query_params):
        if resource.get(key_name) == param_value:
            region_param = self.module.params.get('region')
            region_resource = resource.get('region')
            if region_resource and region_param and (region_param != region_resource):
                continue
            if found:
                if region_resource and (not region_param):
                    msg = 'More than one record with name=%s found. Use region to distinguish.' % param_value
                else:
                    msg = 'More than one record with name=%s found. Use multiple=true if module supports it.' % param_value
                self.module.fail_json(msg=msg)
            found = resource
    if found:
        if get_details:
            return self.query_by_id(resource_id=found[key_id], skip_transform=skip_transform)
        elif skip_transform:
            return found
        else:
            return self.transform_resource(found)
    elif fail_not_found:
        self.module.fail_json(msg='No Resource %s with %s found: %s' % (path, key_name, param_value))
    return dict()