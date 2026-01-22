from __future__ import absolute_import, division, print_function
import copy
import os
import re
import datetime
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import exec_command
from ansible.module_utils.six import iteritems
from ansible.module_utils.parsing.convert_bool import (
from collections import defaultdict
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from .constants import (
def transform_name(partition='', name='', sub_path=''):
    if partition != '':
        if name.startswith(partition + '/'):
            name = name.replace(partition + '/', '')
        if name.startswith('/' + partition + '/'):
            name = name.replace('/' + partition + '/', '')
    if name:
        name = name.replace('/', '~')
        name = name.replace('%', '%25')
    if partition:
        partition = partition.replace('/', '~')
        if not partition.startswith('~'):
            partition = '~' + partition
    elif sub_path:
        raise F5ModuleError('When giving the subPath component include partition as well.')
    if sub_path and partition:
        sub_path = '~' + sub_path
    if name and partition:
        name = '~' + name
    result = partition + sub_path + name
    return result