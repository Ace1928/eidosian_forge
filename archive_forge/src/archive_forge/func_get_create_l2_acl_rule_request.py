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
def get_create_l2_acl_rule_request(self, acl_name, seq_num, rule):
    """Get request to create a rule with given sequence number
        and configuration in the specified L2 ACL
        """
    url = self.l2_acl_rule_path.format(acl_name=acl_name)
    payload = {'openconfig-acl:acl-entry': [{'sequence-id': seq_num, 'config': {'sequence-id': seq_num}, 'l2': {'config': {}}, 'actions': {'config': {'forwarding-action': action_value_to_payload_map[rule['action']]}}}]}
    rule_l2_config = payload['openconfig-acl:acl-entry'][0]['l2']['config']
    if rule['source'].get('host'):
        rule_l2_config['source-mac'] = rule['source']['host']
    elif rule['source'].get('address'):
        rule_l2_config['source-mac'] = rule['source']['address']
        rule_l2_config['source-mac-mask'] = rule['source']['address_mask']
    if rule['destination'].get('host'):
        rule_l2_config['destination-mac'] = rule['destination']['host']
    elif rule['destination'].get('address'):
        rule_l2_config['destination-mac'] = rule['destination']['address']
        rule_l2_config['destination-mac-mask'] = rule['destination']['address_mask']
    if rule.get('ethertype'):
        if rule['ethertype'].get('value'):
            rule_l2_config['ethertype'] = ethertype_value_to_payload_map.get(rule['ethertype']['value'], int(rule['ethertype']['value'], 16))
        else:
            rule_l2_config['ethertype'] = ethertype_protocol_to_payload_map[next(iter(rule['ethertype']))]
    if rule.get('vlan_id') is not None:
        rule_l2_config['vlanid'] = rule['vlan_id']
    if rule.get('vlan_tag_format') and rule['vlan_tag_format'].get('multi_tagged'):
        rule_l2_config['vlan-tag-format'] = 'openconfig-acl-ext:MULTI_TAGGED'
    if rule.get('dei') is not None:
        rule_l2_config['dei'] = rule['dei']
    if rule.get('pcp'):
        if rule['pcp'].get('traffic_type'):
            rule_l2_config['pcp'] = pcp_traffic_to_value_map[rule['pcp']['traffic_type']]
        else:
            rule_l2_config['pcp'] = rule['pcp']['value']
            rule_l2_config['pcp-mask'] = rule['pcp']['mask']
    if rule.get('remark'):
        payload['openconfig-acl:acl-entry'][0]['config']['description'] = rule['remark']
    return {'path': url, 'method': POST, 'data': payload}