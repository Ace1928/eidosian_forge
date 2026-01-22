from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def config_counter(self):
    """configures sflow counter on an interface"""
    xml_str = ''
    if not self.sflow_interface:
        return xml_str
    if not self.sflow_dict['counter'] and self.state == 'absent':
        return xml_str
    self.updates_cmd.append('interface %s' % self.sflow_interface)
    if self.state == 'present':
        xml_str += '<counters><counter operation="merge"><ifName>%s</ifName>' % self.sflow_interface
    else:
        xml_str += '<counters><counter operation="delete"><ifName>%s</ifName>' % self.sflow_interface
    if self.counter_collector:
        if self.sflow_dict['counter'].get('collectorID') and self.sflow_dict['counter'].get('collectorID') != 'invalid':
            existing = self.sflow_dict['counter'].get('collectorID').split(',')
        else:
            existing = list()
        if self.state == 'present':
            diff = list(set(self.counter_collector) - set(existing))
            if diff:
                self.updates_cmd.append('sflow counter collector %s' % ' '.join(diff))
                new_set = list(self.counter_collector + existing)
                xml_str += '<collectorID>%s</collectorID>' % ','.join(list(set(new_set)))
        else:
            same = list(set(self.counter_collector) & set(existing))
            if same:
                self.updates_cmd.append('undo sflow counter collector %s' % ' '.join(same))
                xml_str += '<collectorID>%s</collectorID>' % ','.join(list(set(same)))
    if self.counter_interval:
        exist = bool(self.counter_interval == self.sflow_dict['counter'].get('interval'))
        if self.state == 'present' and (not exist):
            self.updates_cmd.append('sflow counter interval %s' % self.counter_interval)
            xml_str += '<interval>%s</interval>' % self.counter_interval
        elif self.state == 'absent' and exist:
            self.updates_cmd.append('undo sflow counter interval %s' % self.counter_interval)
            xml_str += '<interval>%s</interval>' % self.counter_interval
    if xml_str.endswith('</ifName>'):
        self.updates_cmd.pop()
        return ''
    xml_str += '</counter></counters>'
    return xml_str