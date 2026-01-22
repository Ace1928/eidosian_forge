from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
def construct_rule(params):
    rule = []
    append_wait(rule, params['wait'], '-w')
    append_param(rule, params['protocol'], '-p', False)
    append_param(rule, params['source'], '-s', False)
    append_param(rule, params['destination'], '-d', False)
    append_param(rule, params['match'], '-m', True)
    append_tcp_flags(rule, params['tcp_flags'], '--tcp-flags')
    append_param(rule, params['jump'], '-j', False)
    if params.get('jump') and params['jump'].lower() == 'tee':
        append_param(rule, params['gateway'], '--gateway', False)
    append_param(rule, params['log_prefix'], '--log-prefix', False)
    append_param(rule, params['log_level'], '--log-level', False)
    append_param(rule, params['to_destination'], '--to-destination', False)
    append_match(rule, params['destination_ports'], 'multiport')
    append_csv(rule, params['destination_ports'], '--dports')
    append_param(rule, params['to_source'], '--to-source', False)
    append_param(rule, params['goto'], '-g', False)
    append_param(rule, params['in_interface'], '-i', False)
    append_param(rule, params['out_interface'], '-o', False)
    append_param(rule, params['fragment'], '-f', False)
    append_param(rule, params['set_counters'], '-c', False)
    append_param(rule, params['source_port'], '--source-port', False)
    append_param(rule, params['destination_port'], '--destination-port', False)
    append_param(rule, params['to_ports'], '--to-ports', False)
    append_param(rule, params['set_dscp_mark'], '--set-dscp', False)
    append_param(rule, params['set_dscp_mark_class'], '--set-dscp-class', False)
    append_match_flag(rule, params['syn'], '--syn', True)
    if 'conntrack' in params['match']:
        append_csv(rule, params['ctstate'], '--ctstate')
    elif 'state' in params['match']:
        append_csv(rule, params['ctstate'], '--state')
    elif params['ctstate']:
        append_match(rule, params['ctstate'], 'conntrack')
        append_csv(rule, params['ctstate'], '--ctstate')
    if 'iprange' in params['match']:
        append_param(rule, params['src_range'], '--src-range', False)
        append_param(rule, params['dst_range'], '--dst-range', False)
    elif params['src_range'] or params['dst_range']:
        append_match(rule, params['src_range'] or params['dst_range'], 'iprange')
        append_param(rule, params['src_range'], '--src-range', False)
        append_param(rule, params['dst_range'], '--dst-range', False)
    if 'set' in params['match']:
        append_param(rule, params['match_set'], '--match-set', False)
        append_match_flag(rule, 'match', params['match_set_flags'], False)
    elif params['match_set']:
        append_match(rule, params['match_set'], 'set')
        append_param(rule, params['match_set'], '--match-set', False)
        append_match_flag(rule, 'match', params['match_set_flags'], False)
    append_match(rule, params['limit'] or params['limit_burst'], 'limit')
    append_param(rule, params['limit'], '--limit', False)
    append_param(rule, params['limit_burst'], '--limit-burst', False)
    append_match(rule, params['uid_owner'], 'owner')
    append_match_flag(rule, params['uid_owner'], '--uid-owner', True)
    append_param(rule, params['uid_owner'], '--uid-owner', False)
    append_match(rule, params['gid_owner'], 'owner')
    append_match_flag(rule, params['gid_owner'], '--gid-owner', True)
    append_param(rule, params['gid_owner'], '--gid-owner', False)
    if params['jump'] is None:
        append_jump(rule, params['reject_with'], 'REJECT')
    append_param(rule, params['reject_with'], '--reject-with', False)
    append_param(rule, params['icmp_type'], ICMP_TYPE_OPTIONS[params['ip_version']], False)
    append_match(rule, params['comment'], 'comment')
    append_param(rule, params['comment'], '--comment', False)
    return rule