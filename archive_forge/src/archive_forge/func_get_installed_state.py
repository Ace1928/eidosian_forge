from __future__ import absolute_import, division, print_function
import os.path
import xml
import re
from xml.dom.minidom import parseString as parseXML
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
def get_installed_state(m, packages):
    """get installed state of packages"""
    cmd = get_cmd(m, 'search')
    cmd.extend(['--match-exact', '--details', '--installed-only'])
    cmd.extend([p.name for p in packages])
    return parse_zypper_xml(m, cmd, fail_not_found=False)[0]