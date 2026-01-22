from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.univention_umc import (
def convert_time(time):
    """Convert a time in seconds into the biggest unit"""
    units = [(24 * 60 * 60, 'days'), (60 * 60, 'hours'), (60, 'minutes'), (1, 'seconds')]
    if time == 0:
        return ('0', 'seconds')
    for unit in units:
        if time >= unit[0]:
            return ('{0}'.format(time // unit[0]), unit[1])