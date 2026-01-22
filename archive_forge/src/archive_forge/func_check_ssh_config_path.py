from __future__ import absolute_import, division, print_function
import os
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils._stormssh import ConfigParser, HAS_PARAMIKO, PARAMIKO_IMPORT_ERROR
from ansible_collections.community.general.plugins.module_utils.ssh import determine_config_file
def check_ssh_config_path(self):
    self.config_file = determine_config_file(self.user, self.config_file)
    if os.path.exists(self.config_file) and self.identity_file is not None:
        dirname = os.path.dirname(self.config_file)
        self.identity_file = os.path.join(dirname, self.identity_file)
        if not os.path.exists(self.identity_file):
            self.module.fail_json(msg='IdentityFile %s does not exist' % self.params['identity_file'])