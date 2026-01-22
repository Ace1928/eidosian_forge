from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def get_parent_domain(self, key=None):
    path = self.module.params.get('path')
    path = '/'.join(path.split('/')[:-1])
    if not path:
        return None
    parent_domain = self._get_domain_internal(path=path)
    if not parent_domain:
        self.module.fail_json(msg='Parent domain path %s does not exist' % path)
    return self._get_by_key(key, parent_domain)