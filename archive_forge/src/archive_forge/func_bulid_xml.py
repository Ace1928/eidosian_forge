from __future__ import (absolute_import, division, print_function)
import xml.etree.ElementTree as ET
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def bulid_xml(kwargs, operation='get'):
    """create a xml tree by dictionary with operation,get,merge and delete"""
    attrib = {'xmlns': 'http://www.huawei.com/netconf/vrp', 'content-version': '1.0', 'format-version': '1.0'}
    root = ET.Element('ifmtrunk')
    for key in kwargs.keys():
        if key in ('global_priority',):
            xpath = 'lacpSysInfo'
        elif key in ('priority',):
            xpath = 'TrunkIfs/TrunkIf/TrunkMemberIfs/TrunkMemberIf/lacpPortInfo/lacpPort'
        elif key in ['preempt_enable', 'timeout_type', 'fast_timeout', 'select', 'preempt_delay', 'max_active_linknumber', 'collector_delay', 'mixed_rate_link_enable', 'state_flapping', 'unexpected_mac_disable', 'system_id', 'port_id_extension_enable']:
            xpath = 'TrunkIfs/TrunkIf/lacpTrunk'
        elif key in ('trunk_id', 'mode'):
            xpath = 'TrunkIfs/TrunkIf'
        if xpath != '':
            parent = has_element(root, xpath)
            element = ET.SubElement(parent, LACP[key])
            if operation == 'merge':
                parent.attrib = dict(operation=operation)
                element.text = str(kwargs[key])
            if key == 'mode':
                element.text = str(kwargs[key])
            if key == 'trunk_id':
                element.text = 'Eth-Trunk' + str(kwargs[key])
    root.attrib = attrib
    config = ET.tostring(root)
    if operation == 'merge' or operation == 'delete':
        return '<config>%s</config>' % to_native(config)
    return '<filter type="subtree">%s</filter>' % to_native(config)