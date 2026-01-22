from __future__ import absolute_import, division, print_function
import logging
from decimal import Decimal
import re
import traceback
import math
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell.logging_handler \
from ansible.module_utils.basic import missing_required_lib
def is_initiator_valid(value):
    """Validate format of the FC or iSCSI initiator"""
    if value.startswith('iqn') or re.match('([A-Fa-f0-9]{2}:){15}[A-Fa-f0-9]{2}', value, re.I) is not None:
        return True
    else:
        return False