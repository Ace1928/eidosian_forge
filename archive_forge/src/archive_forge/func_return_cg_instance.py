from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def return_cg_instance(self, cg_name):
    """Return the consistency group instance.
            :param cg_name: The name of the consistency group
            :return: Instance of the consistency group
        """
    try:
        cg_details = self.unity_conn.get_cg(name=cg_name)
        cg_id = cg_details.get_id()
        cg_obj = utils.cg.UnityConsistencyGroup.get(self.unity_conn._cli, cg_id)
        return cg_obj
    except Exception as e:
        msg = 'Failed to get the consistency group {0} instance with error {1}'.format(cg_name, str(e))
        LOG.error(msg)
        self.module.fail_json(msg=msg)