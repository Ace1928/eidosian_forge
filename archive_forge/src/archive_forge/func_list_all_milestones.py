from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
from datetime import datetime
def list_all_milestones(self):
    page_nb = 1
    milestones = []
    vars_page = self.gitlab_object.milestones.list(page=page_nb)
    while len(vars_page) > 0:
        milestones += vars_page
        page_nb += 1
        vars_page = self.gitlab_object.milestones.list(page=page_nb)
    return milestones