from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.six.moves.urllib import parse as urllib_parse
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.rabbitmq.plugins.module_utils.rabbitmq import rabbitmq_argument_spec
def change_required(self):
    """
        :return:
        """
    if self.module.params['state'] == 'present':
        if not self.is_present():
            return True
    elif self.module.params['state'] == 'absent':
        if self.is_present():
            return True
    return False