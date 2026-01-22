from __future__ import absolute_import, division, print_function
import os
def determine_config_file(user, config_file):
    if user:
        config_file = os.path.join(os.path.expanduser('~%s' % user), '.ssh', 'config')
    elif config_file is None:
        config_file = '/etc/ssh/ssh_config'
    return config_file