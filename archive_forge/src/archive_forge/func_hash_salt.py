from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
def hash_salt(password):
    split_password = password.split('$')
    if len(split_password) != 4:
        _raise_error('Could not parse salt out password correctly from {0}'.format(password))
    else:
        return split_password[2]