from __future__ import absolute_import, division, print_function
import os.path
import xml
import re
from xml.dom.minidom import parseString as parseXML
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
def split_name_version(name):
    """splits of the package name and desired version

    example formats:
        - docker>=1.10
        - apache=2.4

    Allowed version specifiers: <, >, <=, >=, =
    Allowed version format: [0-9.-]*

    Also allows a prefix indicating remove "-", "~" or install "+"
    """
    prefix = ''
    if name[0] in ['-', '~', '+']:
        prefix = name[0]
        name = name[1:]
    if prefix == '~':
        prefix = '-'
    version_check = re.compile('^(.*?)((?:<|>|<=|>=|=)[0-9.-]*)?$')
    try:
        reres = version_check.match(name)
        name, version = reres.groups()
        if version is None:
            version = ''
        return (prefix, name, version)
    except Exception:
        return (prefix, name, '')