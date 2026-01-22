from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def get_mr(self, title, source_branch, target_branch, state_filter):
    mrs = []
    try:
        mrs = self.project.mergerequests.list(search=title, source_branch=source_branch, target_branch=target_branch, state=state_filter)
    except gitlab.exceptions.GitlabGetError as e:
        self._module.fail_json(msg='Failed to list the Merge Request: %s' % to_native(e))
    if len(mrs) > 1:
        self._module.fail_json(msg='Multiple Merge Requests matched search criteria.')
    if len(mrs) == 1:
        try:
            return self.project.mergerequests.get(id=mrs[0].iid)
        except gitlab.exceptions.GitlabGetError as e:
            self._module.fail_json(msg='Failed to get the Merge Request: %s' % to_native(e))