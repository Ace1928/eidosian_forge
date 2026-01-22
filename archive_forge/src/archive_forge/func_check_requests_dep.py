from __future__ import absolute_import, division, print_function
import json
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six import PY3
from ansible.module_utils.common.text.converters import to_native
def check_requests_dep(module):
    """Check if an adequate requests version is available"""
    if not HAS_REQUESTS:
        module.fail_json(msg=missing_required_lib('requests'), exception=REQUESTS_IMP_ERR)
    else:
        required_version = '2.0.0' if PY3 else '1.0.0'
        if LooseVersion(requests.__version__) < LooseVersion(required_version):
            module.fail_json(msg="'requests' library version should be >= %s, found: %s." % (required_version, requests.__version__))