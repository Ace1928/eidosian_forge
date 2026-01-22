from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.snmp_server.snmp_server import (
def get_alarms(self, cfg):
    cfg_dict = {}
    cfg_dict['id'] = cfg['name']
    if 'description' in cfg.keys():
        cfg_dict['description'] = cfg.get('description')
    if 'falling-event-index' in cfg.keys():
        cfg_dict['falling_event_index'] = cfg.get('falling-event-index')
    if 'falling-threshold' in cfg.keys():
        cfg_dict['falling_threshold'] = cfg.get('falling-threshold')
    if 'falling-threshold-interval' in cfg.keys():
        cfg_dict['falling_threshold_interval'] = cfg.get('falling-threshold-interval')
    if 'interval' in cfg.keys():
        cfg_dict['interval'] = cfg.get('interval')
    if 'request-type' in cfg.keys():
        cfg_dict['request_type'] = cfg.get('request-type')
    if 'rising-event-index' in cfg.keys():
        cfg_dict['rising_event_index'] = cfg.get('rising-event-index')
    if 'rising-threshold' in cfg.keys():
        cfg_dict['rising_threshold'] = cfg.get('rising-threshold')
    if 'sample-type' in cfg.keys():
        cfg_dict['sample_type'] = cfg.get('sample-type')
    if 'startup-alarm' in cfg.keys():
        cfg_dict['startup_alarm'] = cfg.get('startup-alarm')
    if 'syslog-subtag' in cfg.keys():
        cfg_dict['syslog_subtag'] = cfg.get('syslog-subtag')
    if 'variable' in cfg.keys():
        cfg_dict['variable'] = cfg.get('variable')
    return cfg_dict