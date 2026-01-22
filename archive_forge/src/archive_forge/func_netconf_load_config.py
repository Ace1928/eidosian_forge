from __future__ import absolute_import, division, print_function
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def netconf_load_config(self, xml_str):
    """load isis config by netconf"""
    if not xml_str:
        return
    xml_cfg = '\n            <config>\n            <isiscomm xmlns="http://www.huawei.com/netconf/vrp" content-version="1.0" format-version="1.0">\n            %s\n            </isiscomm>\n            </config>' % xml_str
    set_nc_config(self.module, xml_cfg)
    self.changed = True