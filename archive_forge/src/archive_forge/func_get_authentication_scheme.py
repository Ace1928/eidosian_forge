from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_authentication_scheme(self, **kwargs):
    """ Get scheme of authentication """
    module = kwargs['module']
    conf_str = CE_GET_AUTHENTICATION_SCHEME
    xml_str = self.netconf_get_config(module=module, conf_str=conf_str)
    result = list()
    if '<data/>' in xml_str:
        return result
    else:
        re_find = re.findall('.*<authenSchemeName>(.*)</authenSchemeName>.*\\s*<firstAuthenMode>(.*)</firstAuthenMode>.*\\s*<secondAuthenMode>(.*)</secondAuthenMode>.*\\s*', xml_str)
        if re_find:
            return re_find
        else:
            return result