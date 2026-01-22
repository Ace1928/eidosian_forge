from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_current_uds_enum(self, current_uds):
    """
        Get the enum of the Offline Availability parameter.
        :param current_uds: Current Unix Directory Service string
        :return: current_uds enum
        """
    if current_uds in utils.NasServerUnixDirectoryServiceEnum.__members__:
        return utils.NasServerUnixDirectoryServiceEnum[current_uds]
    else:
        error_msg = 'Invalid value {0} for Current Unix Directory Service provided'.format(current_uds)
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)