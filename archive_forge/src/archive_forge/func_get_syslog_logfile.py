from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config, check_ip_addr
def get_syslog_logfile(self):
    """get syslog logfile"""
    cur_logfile_info = dict()
    conf_str = CE_NC_GET_LOG_FILE_INFO_HEADER
    conf_str += '<logFileType>log</logFileType>'
    if self.logfile_max_num:
        conf_str += '<maxFileNum></maxFileNum>'
    if self.logfile_max_size:
        conf_str += '<maxFileSize></maxFileSize>'
    conf_str += CE_NC_GET_LOG_FILE_INFO_TAIL
    xml_str = get_nc_config(self.module, conf_str)
    if '<data/>' in xml_str:
        return cur_logfile_info
    else:
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        logfile_info = root.findall('syslog/icLogFileInfos/icLogFileInfo')
        if logfile_info:
            for tmp in logfile_info:
                for site in tmp:
                    if site.tag in ['maxFileNum', 'maxFileSize']:
                        cur_logfile_info[site.tag] = site.text
        return cur_logfile_info