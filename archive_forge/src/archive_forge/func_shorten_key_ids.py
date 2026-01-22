from __future__ import absolute_import, division, print_function
import os
from traceback import format_exc
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.urls import fetch_url
def shorten_key_ids(key_id_list):
    """
    Takes a list of key ids, and converts them to the 'short' format,
    by reducing them to their last 8 characters.
    """
    short = []
    for key in key_id_list:
        short.append(key[-8:])
    return short