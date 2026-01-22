from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
@staticmethod
def is_repeated_password(message):
    return message.startswith('New password must be different than last 6 passwords.') or message.startswith('New password must be different from last 6 passwords.') or message.startswith('New password must be different than the old password.') or message.startswith('New password must be different from the old password.')