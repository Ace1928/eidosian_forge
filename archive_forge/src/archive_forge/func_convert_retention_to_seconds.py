from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def convert_retention_to_seconds(desired_retention, retention_unit):
    """Convert desired retention to seconds.
        :param desired_retention: The desired retention for snapshot
         schedule
        :param retention_unit: The retention unit for snapshot schedule
        :return: The integer value in seconds
    """
    duration_in_sec = None
    if desired_retention:
        if retention_unit == 'hours':
            duration_in_sec = desired_retention * 60 * 60
        else:
            duration_in_sec = desired_retention * 24 * 60 * 60
    return duration_in_sec