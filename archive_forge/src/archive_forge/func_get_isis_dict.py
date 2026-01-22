from __future__ import absolute_import, division, print_function
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def get_isis_dict(self):
    """isis config dict"""
    isis_dict = dict()
    isis_dict['instance'] = dict()
    conf_str = CE_NC_GET_ISIS % (CE_NC_GET_ISIS_INSTANCE % self.instance_id)
    xml_str = get_nc_config(self.module, conf_str)
    if '<data/>' in xml_str:
        return isis_dict
    xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    glb = root.find('isiscomm/isSites/isSite')
    if glb:
        for attr in glb:
            isis_dict['instance'][attr.tag] = attr.text
    return isis_dict