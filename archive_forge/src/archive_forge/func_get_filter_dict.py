from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config, check_ip_addr
def get_filter_dict(self):
    """ get syslog filter attributes dict."""
    filter_info = dict()
    conf_str = CE_NC_GET_FILTER_INFO
    xml_str = get_nc_config(self.module, conf_str)
    if '<data/>' in xml_str:
        return filter_info
    xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    filter_info['filterInfos'] = list()
    ic_filters = root.findall('syslog/icFilters/icFilter')
    if ic_filters:
        for ic_filter in ic_filters:
            filter_dict = dict()
            for ele in ic_filter:
                if ele.tag in ['icFeatureName', 'icFilterLogName']:
                    filter_dict[ele.tag] = ele.text
            filter_info['filterInfos'].append(filter_dict)
    return filter_info