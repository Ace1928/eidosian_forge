from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def get_number_and_section_from_relative_position(payload, connection, version, rulebase, above_relative_position, pos_before_relative_empty_section, api_call_object, prev_section=None, current_section=None):
    section_name = current_section
    position = None
    for rules in rulebase:
        if 'rulebase' in rules:
            if 'above' in payload['position'] and rules['name'] == payload['position']['above']:
                if len(rules['rulebase']) == 0:
                    position = pos_before_relative_empty_section if above_relative_position else pos_before_relative_empty_section + 1
                else:
                    from_value = get_edge_position_in_section(connection, version, list(get_relevant_layer_or_package_identifier(api_call_object, payload).values())[0], rules['name'], 'from')
                    if from_value is not None:
                        position = max(from_value - 1, 1) if above_relative_position else from_value
                return (position, section_name, above_relative_position, pos_before_relative_empty_section, prev_section)
            prev_section = section_name
            section_name = rules['name']
            if 'bottom' in payload['position'] and rules['name'] == payload['position']['bottom']:
                if len(rules['rulebase']) == 0:
                    position = pos_before_relative_empty_section if above_relative_position else pos_before_relative_empty_section + 1
                else:
                    to_value = get_edge_position_in_section(connection, version, list(get_relevant_layer_or_package_identifier(api_call_object, payload).values())[0], section_name, 'to')
                    if to_value is not None and to_value == int(rules['to']):
                        is_bottom = rules['rulebase'][-1]['name'] == payload['name']
                        position = to_value if above_relative_position or is_bottom else to_value + 1
                return (position, section_name, above_relative_position, pos_before_relative_empty_section, prev_section)
            if 'below' in payload['position'] and section_name == payload['position']['below'] or ('top' in payload['position'] and section_name == payload['position']['top']):
                if len(rules['rulebase']) == 0:
                    position = pos_before_relative_empty_section if above_relative_position else pos_before_relative_empty_section + 1
                else:
                    is_top = rules['rulebase'][0]['name'] == payload['name']
                    position = max(int(rules['from']) - 1, 1) if above_relative_position or not is_top else int(rules['from'])
                return (position, section_name, above_relative_position, pos_before_relative_empty_section, prev_section)
            if len(rules['rulebase']) != 0:
                pos_before_relative_empty_section = int(rules['to'])
            rules = rules['rulebase']
            for rule in rules:
                if payload['name'] == rule['name']:
                    above_relative_position = True
                if 'below' in payload['position'] and rule['name'] == payload['position']['below']:
                    position = int(rule['rule-number']) if above_relative_position else int(rule['rule-number']) + 1
                    return (position, section_name, above_relative_position, pos_before_relative_empty_section, prev_section)
                elif 'above' in payload['position'] and rule['name'] == payload['position']['above']:
                    position = max(int(rule['rule-number']) - 1, 1) if above_relative_position else int(rule['rule-number'])
                    return (position, section_name, above_relative_position, pos_before_relative_empty_section, prev_section)
        else:
            if payload['name'] == rules['name']:
                above_relative_position = True
            if 'below' in payload['position'] and rules['name'] == payload['position']['below']:
                position = int(rules['rule-number']) if above_relative_position else int(rules['rule-number']) + 1
                return (position, section_name, above_relative_position, pos_before_relative_empty_section, prev_section)
            elif 'above' in payload['position'] and rules['name'] == payload['position']['above']:
                position = max(int(rules['rule-number']) - 1, 1) if above_relative_position else int(rules['rule-number'])
                return (position, section_name, above_relative_position, pos_before_relative_empty_section, prev_section)
    return (position, section_name, above_relative_position, pos_before_relative_empty_section, prev_section)