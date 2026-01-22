from __future__ import absolute_import, division, print_function
import os
from traceback import format_exc
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.urls import fetch_url
def find_needed_binaries(module):
    global apt_key_bin
    global gpg_bin
    apt_key_bin = module.get_bin_path('apt-key', required=True)
    gpg_bin = module.get_bin_path('gpg', required=True)