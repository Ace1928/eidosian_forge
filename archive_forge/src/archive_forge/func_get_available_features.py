from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_available_features(feature, module):
    available_features = {}
    feature_regex = '(?P<feature>\\S+)\\s+\\d+\\s+(?P<state>.*)'
    command = {'command': 'show feature', 'output': 'text'}
    try:
        body = run_commands(module, [command])[0]
        split_body = body.splitlines()
    except (KeyError, IndexError):
        return {}
    for line in split_body:
        try:
            match_feature = re.match(feature_regex, line, re.DOTALL)
            feature_group = match_feature.groupdict()
            feature = feature_group['feature']
            state = feature_group['state']
        except AttributeError:
            feature = ''
            state = ''
        if feature and state:
            if 'enabled' in state:
                state = 'enabled'
            if feature not in available_features:
                available_features[feature] = state
            elif available_features[feature] == 'disabled' and state == 'enabled':
                available_features[feature] = state
    run_cfg = get_config(module, flags=['| include ^feature'])
    for item in re.findall('feature\\s(.*)', run_cfg):
        if item not in available_features:
            available_features[item] = 'enabled'
    if 'fabric forwarding' not in available_features:
        available_features['fabric forwarding'] = 'disabled'
    return available_features