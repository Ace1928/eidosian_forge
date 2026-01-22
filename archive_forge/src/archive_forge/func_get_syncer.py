from __future__ import (absolute_import, division, print_function)
import traceback
from datetime import datetime
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.okd.plugins.module_utils.openshift_ldap import (
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def get_syncer(self):
    syncer = None
    if 'rfc2307' in self.config:
        syncer = OpenshiftLDAPRFC2307(self.config, self.connection)
    elif 'activeDirectory' in self.config:
        syncer = OpenshiftLDAPActiveDirectory(self.config, self.connection)
    elif 'augmentedActiveDirectory' in self.config:
        syncer = OpenshiftLDAPAugmentedActiveDirectory(self.config, self.connection)
    else:
        msg = "No schema-specific config was found, should be one of 'rfc2307', 'activeDirectory', 'augmentedActiveDirectory'"
        self.fail_json(msg=msg)
    return syncer