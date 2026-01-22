from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def mr_has_changed(self, mr, options):
    for key, value in options.items():
        if value is not None:
            if key == 'remove_source_branch':
                key = 'force_remove_source_branch'
            if key == 'assignee_ids':
                if options[key] != sorted([user['id'] for user in getattr(mr, 'assignees')]):
                    return True
            elif key == 'reviewer_ids':
                if options[key] != sorted([user['id'] for user in getattr(mr, 'reviewers')]):
                    return True
            elif key == 'labels':
                if options[key] != sorted(getattr(mr, key)):
                    return True
            elif getattr(mr, key) != value:
                return True
    return False