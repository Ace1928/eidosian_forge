from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def file_interface_result_list(entity):
    """ Get the id, name and IP of File Interfaces """
    result = []
    if entity:
        LOG.info(SUCCESSFULL_LISTED_MSG)
        for item in entity:
            result.append(item._get_properties())
        return result
    else:
        return None