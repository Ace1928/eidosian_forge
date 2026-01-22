from __future__ import (absolute_import, division, print_function)
import xml.etree.ElementTree as ET
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def xml_to_dict(args):
    """transfer xml string into dict """
    rdict = dict()
    args = re.sub('xmlns=\\".+?\\"', '', args)
    root = ET.fromstring(args)
    ifmtrunk = root.find('.//ifmtrunk')
    if ifmtrunk is not None:
        try:
            ifmtrunk_iter = ET.Element.iter(ifmtrunk)
        except AttributeError:
            ifmtrunk_iter = ifmtrunk.getiterator()
        for ele in ifmtrunk_iter:
            if ele.text is not None and len(ele.text.strip()) > 0:
                rdict[ele.tag] = ele.text
    return rdict