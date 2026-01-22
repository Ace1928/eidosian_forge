from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import WapiModule
from ..module_utils.api import NIOS_NSGROUP
from ..module_utils.api import normalize_ib_spec
def clean_tsig(ext):
    if 'tsig_key' in ext and (not ext['tsig_key']):
        del ext['tsig_key']
    if 'tsig_key' not in ext and 'tsig_key_name' in ext and (not ext['tsig_key_name']):
        del ext['tsig_key_name']
    if 'tsig_key' not in ext and 'tsig_key_alg' in ext:
        del ext['tsig_key_alg']