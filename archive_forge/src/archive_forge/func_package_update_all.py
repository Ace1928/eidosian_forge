from __future__ import absolute_import, division, print_function
import os.path
import xml
import re
from xml.dom.minidom import parseString as parseXML
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
def package_update_all(m):
    """run update or patch on all available packages"""
    retvals = {'rc': 0, 'stdout': '', 'stderr': ''}
    if m.params['type'] == 'patch':
        cmdname = 'patch'
    elif m.params['state'] == 'dist-upgrade':
        cmdname = 'dist-upgrade'
    else:
        cmdname = 'update'
    cmd = get_cmd(m, cmdname)
    retvals['cmd'] = cmd
    result, retvals['rc'], retvals['stdout'], retvals['stderr'] = parse_zypper_xml(m, cmd)
    return (result, retvals)