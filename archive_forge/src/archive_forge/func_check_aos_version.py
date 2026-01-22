from ansible.module_utils.network.aos.aos import (check_aos_version, get_aos_session, find_collection_item,
from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.network.plugins.module_utils.version import LooseVersion
from ansible.module_utils._text import to_native
def check_aos_version(module, min=False):
    """
    Check if the library aos-pyez is present.
    If provided, also check if the minimum version requirement is met
    """
    if not HAS_AOS_PYEZ:
        module.fail_json(msg='aos-pyez is not installed.  Please see details here: https://github.com/Apstra/aos-pyez')
    elif min:
        import apstra.aosom
        AOS_PYEZ_VERSION = apstra.aosom.__version__
        if LooseVersion(AOS_PYEZ_VERSION) < LooseVersion(min):
            module.fail_json(msg='aos-pyez >= %s is required for this module' % min)
    return True