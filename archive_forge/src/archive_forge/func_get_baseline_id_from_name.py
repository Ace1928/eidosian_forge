from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_baseline_id_from_name(rest_obj, module):
    try:
        baseline_name = module.params.get('baseline_name')
        baseline_id = 0
        if baseline_name is not None:
            resp_val = rest_obj.get_all_items_with_pagination(base_line_path)
            baseline_list = resp_val['value']
            if baseline_list:
                for baseline in baseline_list:
                    if baseline['Name'] == baseline_name:
                        baseline_id = baseline['Id']
                        break
                else:
                    module.exit_json(msg='Specified baseline_name does not exist in the system.', baseline_compliance_info=[])
            else:
                module.exit_json(msg='No baseline exists in the system.', baseline_compliance_info=[])
        else:
            module.fail_json(msg='baseline_name is a mandatory option.')
        return baseline_id
    except (URLError, HTTPError, SSLValidationError, ConnectionError, TypeError, ValueError) as err:
        raise err