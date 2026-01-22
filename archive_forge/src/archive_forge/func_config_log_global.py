from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def config_log_global(self):
    """config log global param"""
    xml_str = '<globalParam operation="merge">'
    if self.log_time_stamp:
        if self.state == 'present' and self.log_time_stamp.upper() != self.log_dict.get('logTimeStamp'):
            xml_str += '<logTimeStamp>%s</logTimeStamp>' % self.log_time_stamp.upper()
            self.updates_cmd.append('info-center timestamp log %s' % TIME_STAMP_DICT.get(self.log_time_stamp))
        elif self.state == 'absent' and self.log_time_stamp.upper() == self.log_dict.get('logTimeStamp'):
            xml_str += '<logTimeStamp>DATE_SECOND</logTimeStamp>'
            self.updates_cmd.append('undo info-center timestamp log')
        else:
            pass
    if self.log_buff_enable != 'no_use':
        if self.log_dict.get('icLogBuffEn') != self.log_buff_enable:
            xml_str += '<icLogBuffEn>%s</icLogBuffEn>' % self.log_buff_enable
            if self.log_buff_enable == 'true':
                self.updates_cmd.append('info-center logbuffer')
            else:
                self.updates_cmd.append('undo info-center logbuffer')
    if self.log_buff_size:
        if self.state == 'present' and self.log_dict.get('bufferSize') != self.log_buff_size:
            xml_str += '<bufferSize>%s</bufferSize>' % self.log_buff_size
            self.updates_cmd.append('info-center logbuffer size %s' % self.log_buff_size)
        elif self.state == 'absent' and self.log_dict.get('bufferSize') == self.log_buff_size:
            xml_str += '<bufferSize>512</bufferSize>'
            self.updates_cmd.append('undo info-center logbuffer size')
    if xml_str == '<globalParam operation="merge">':
        return ''
    else:
        xml_str += '</globalParam>'
        return xml_str