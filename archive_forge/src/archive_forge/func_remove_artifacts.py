from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils._text import to_bytes, to_text
def remove_artifacts(module, client):
    try:
        client.remove_service()
    except (SMBException, PypsexecException) as exc:
        module.warn('Failed to cleanup PAExec service and executable: %s' % to_text(exc))