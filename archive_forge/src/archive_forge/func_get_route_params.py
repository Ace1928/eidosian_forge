from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
def get_route_params(self, raw_values):
    routes_params = []
    for raw_value in raw_values:
        route_params = {}
        for parameter, value in re.findall('([\\w-]*)\\s?=\\s?([^\\s,}]*)', raw_value):
            if parameter == 'nh':
                route_params['next_hop'] = value
            elif parameter == 'mt':
                route_params['metric'] = value
            else:
                route_params[parameter] = value
        routes_params.append(route_params)
    return [self.route_to_string(route_params) for route_params in routes_params]