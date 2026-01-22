from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def validate_rule(self, rule, rule_type=None):
    """
    Apply defaults to a rule dictionary and check that all values are valid.

    :param rule: rule dict
    :param rule_type: Set to 'default' if the rule is part of the default set of rules.
    :return: None
    """
    priority = rule.get('priority', 0)
    if rule_type != 'default' and (priority < 100 or priority > 4096):
        raise Exception('Rule priority must be between 100 and 4096')

    def check_plural(src, dest):
        if isinstance(rule.get(src), list):
            rule[dest] = rule[src]
            rule[src] = None
    check_plural('destination_address_prefix', 'destination_address_prefixes')
    check_plural('source_address_prefix', 'source_address_prefixes')
    check_plural('source_port_range', 'source_port_ranges')
    check_plural('destination_port_range', 'destination_port_ranges')
    if rule.get('source_application_security_groups') and rule.get('source_address_prefix') == '*':
        rule['source_address_prefix'] = None
    if rule.get('destination_application_security_groups') and rule.get('destination_address_prefix') == '*':
        rule['destination_address_prefix'] = None