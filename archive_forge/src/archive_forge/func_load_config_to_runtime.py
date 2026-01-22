from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.six.moves import configparser
from ansible.module_utils.basic import missing_required_lib
def load_config_to_runtime(cursor, save_what, variable=None):
    if variable and variable.startswith('admin'):
        config_type = 'ADMIN'
    elif save_what == 'SCHEDULER':
        config_type = ''
    else:
        config_type = 'MYSQL'
    cursor.execute('LOAD {0} {1} TO RUNTIME'.format(config_type, save_what))
    return True