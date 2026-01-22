from __future__ import (absolute_import, division, print_function)
import re
import platform
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule

    The ss_parse result can be either split in 6 or 7 elements depending on the process column,
    e.g. due to unprivileged user.
    :param raw: ss raw output String. First line explains the format, each following line contains a connection.
    :return: List of dicts, each dict contains protocol, state, local address, foreign address, port, name, pid for one
     connection.
    