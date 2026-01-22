from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def state_delete_tag(self):
    """
        Delete tag

        """
    tag_id = self.tag_obj.id
    try:
        self.tag_service.delete(tag_id=tag_id)
    except Error as error:
        self.module.fail_json(msg='%s' % self.get_error_message(error))
    self.module.exit_json(changed=True, tag_status=dict(msg="Tag '%s' deleted." % self.tag_name, tag_id=tag_id))