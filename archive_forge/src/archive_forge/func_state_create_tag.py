from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def state_create_tag(self):
    """
        Create tag

        """
    tag_spec = self.tag_service.CreateSpec()
    tag_spec.name = self.tag_name
    tag_spec.description = self.params.get('tag_description')
    '\n        There is no need to check if a category with the specified category_id\n        exists. The tag service will do the corresponding checks and will fail\n        if someone tries to create a tag for a category id that does not exist.\n\n        '
    tag_spec.category_id = self.category_id
    tag_id = ''
    try:
        tag_id = self.tag_service.create(tag_spec)
    except Error as error:
        self.module.fail_json(msg='%s' % self.get_error_message(error))
    if tag_id is not None:
        self.module.exit_json(changed=True, tag_status=dict(msg="Tag '%s' created." % tag_spec.name, tag_id=tag_id))
    self.module.exit_json(changed=False, tag_status=dict(msg='No tag created', tag_id=tag_id))