from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_offline_availability_enum(self, offline_availability):
    """
        Get the enum of the Offline Availability parameter.
        :param offline_availability: The offline_availability string
        :return: offline_availability enum
        """
    if offline_availability in utils.CifsShareOfflineAvailabilityEnum.__members__:
        return utils.CifsShareOfflineAvailabilityEnum[offline_availability]
    else:
        error_msg = 'Invalid value {0} for offline availability provided'.format(offline_availability)
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)