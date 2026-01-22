import os
from ansible.errors import AnsibleAction
from ansible.errors import AnsibleActionFail
from ansible.errors import AnsibleError
from ansible.errors import AnsibleFileNotFound
from ansible.module_utils._text import to_text
from ansible.plugins.action import ActionBase
from ansible.utils.vars import merge_hash
handler for s3_object operations

        This adds the magic that means 'src' can point to both a 'remote' file
        on the 'host' or in the 'files/' lookup path on the controller.
        