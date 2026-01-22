from __future__ import absolute_import, division, print_function
import logging
import math
import re
from decimal import Decimal
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.logging_handler \
import traceback
from ansible.module_utils.basic import missing_required_lib
def ensure_required_libs(module):
    """Check required libraries"""
    if not HAS_DATEUTIL:
        module.fail_json(msg=missing_required_lib('python-dateutil'), exception=DATEUTIL_IMP_ERR)
    if not PKG_RSRC_IMPORTED:
        module.fail_json(msg=missing_required_lib('pkg_resources'), exception=PKG_RSRC_IMP_ERR)
    if not HAS_POWERFLEX_SDK:
        module.fail_json(msg=missing_required_lib('PyPowerFlex V 1.9.0 or above'), exception=POWERFLEX_SDK_IMP_ERR)
    min_ver = '1.9.0'
    try:
        curr_version = pkg_resources.require('PyPowerFlex')[0].version
        supported_version = parse_version(curr_version) >= parse_version(min_ver)
        if not supported_version:
            module.fail_json(msg='PyPowerFlex {0} is not supported. Required minimum version is {1}'.format(curr_version, min_ver))
    except Exception as e:
        module.fail_json(msg='Getting PyPowerFlex SDK version, failed with Error {0}'.format(str(e)))