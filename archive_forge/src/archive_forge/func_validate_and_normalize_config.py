from __future__ import absolute_import, division, print_function
from ast import literal_eval
from ansible.module_utils._text import to_text
from ansible.module_utils.common.validation import check_required_arguments
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def validate_and_normalize_config(self, config_list):
    """Validate and normalize the given config"""
    updated_config_list = [remove_empties(config) for config in config_list]
    validate_config(self._module.argument_spec, {'config': updated_config_list})
    state = self._module.params['state']
    if state == 'deleted':
        return updated_config_list
    for config in updated_config_list:
        if not config.get('acls'):
            continue
        acl_type = config['address_family']
        for acl in config['acls']:
            if not acl.get('rules'):
                continue
            acl_name = acl['name']
            for rule in acl['rules']:
                seq_num = rule['sequence_num']
                self._check_required(['action', 'source', 'destination', 'protocol'], rule, ['config', 'acls', 'rules'])
                self._validate_and_normalize_protocol(acl_type, acl_name, rule)
                protocol = rule['protocol']['name'] if rule['protocol'].get('name') else str(rule['protocol']['number'])
                for endpoint in ('source', 'destination'):
                    if rule[endpoint].get('any') is False:
                        self._invalid_rule('True is the only valid value for {0} -> any'.format(endpoint), acl_type, acl_name, seq_num)
                    elif rule[endpoint].get('host'):
                        rule[endpoint]['host'] = rule[endpoint]['host'].lower()
                    elif rule[endpoint].get('prefix'):
                        rule[endpoint]['prefix'] = rule[endpoint]['prefix'].lower()
                    if rule[endpoint].get('port_number'):
                        if protocol not in ('tcp', 'udp'):
                            self._invalid_rule('{0} -> port_number is valid only for TCP or UDP protocol'.format(endpoint), acl_type, acl_name, seq_num)
                        self._validate_and_normalize_port_number(acl_type, acl_name, rule, endpoint)
                if rule.get('protocol_options'):
                    protocol_options = next(iter(rule['protocol_options']))
                    if protocol != protocol_options:
                        self._invalid_rule('protocol_options -> {0} is not valid for protocol {1}'.format(protocol_options, protocol), acl_type, acl_name, seq_num)
                    self._normalize_protocol_options(rule)
                self._normalize_dscp(rule)
    return updated_config_list