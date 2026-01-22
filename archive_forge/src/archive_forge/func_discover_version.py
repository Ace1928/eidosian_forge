from __future__ import absolute_import, division, print_function
import csv
import socket
import time
from string import Template
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_text
def discover_version(self):
    """
        Attempt to extract the haproxy version.
        Return a tuple containing major and minor version.
        """
    data = self.execute('show info', 200, False)
    lines = data.splitlines()
    line = [x for x in lines if 'Version:' in x]
    try:
        version_values = line[0].partition(':')[2].strip().split('.', 3)
        version = (int(version_values[0]), int(version_values[1]))
    except (ValueError, TypeError, IndexError):
        version = None
    return version