from __future__ import (absolute_import, division, print_function)
import json
import os
from functools import partial
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.validation import check_type_dict, safe_eval
def get_network_view(self, proposed_object):
    """ Check for the associated network view with
            the given dns_view"""
    try:
        network_view_ref = self.get_object('view', {'name': proposed_object['view']}, return_fields=['network_view'])
        if network_view_ref:
            network_view = network_view_ref[0].get('network_view')
    except Exception:
        raise Exception('object with dns_view: %s not found' % proposed_object['view'])
    return network_view