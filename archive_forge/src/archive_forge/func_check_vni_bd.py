from __future__ import (absolute_import, division, print_function)
import re
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def check_vni_bd(self):
    """Check whether vxlan vni is configured in BD view"""
    xml_str = CE_NC_GET_VNI_BD
    xml_str = get_nc_config(self.module, xml_str)
    if '<data/>' in xml_str or not re.findall('<vniId>\\S+</vniId>\\s+<bdId>%s</bdId>' % self.bridge_domain_id, xml_str):
        self.module.fail_json(msg='Error: The vxlan vni is not configured or the bridge domain id is invalid.')