from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_tiering_policy_enum(self, tiering_policy):
    """Get the tiering_policy enum.
             :param tiering_policy: The tiering_policy string
             :return: tiering_policy enum
        """
    if tiering_policy in utils.TieringPolicyEnum.__members__:
        return utils.TieringPolicyEnum[tiering_policy]
    else:
        errormsg = 'Invalid choice {0} for tiering policy'.format(tiering_policy)
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)