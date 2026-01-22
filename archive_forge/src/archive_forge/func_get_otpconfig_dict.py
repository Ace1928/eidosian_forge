from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def get_otpconfig_dict(ipatokentotpauthwindow=None, ipatokentotpsyncwindow=None, ipatokenhotpauthwindow=None, ipatokenhotpsyncwindow=None):
    config = {}
    if ipatokentotpauthwindow is not None:
        config['ipatokentotpauthwindow'] = str(ipatokentotpauthwindow)
    if ipatokentotpsyncwindow is not None:
        config['ipatokentotpsyncwindow'] = str(ipatokentotpsyncwindow)
    if ipatokenhotpauthwindow is not None:
        config['ipatokenhotpauthwindow'] = str(ipatokenhotpauthwindow)
    if ipatokenhotpsyncwindow is not None:
        config['ipatokenhotpsyncwindow'] = str(ipatokenhotpsyncwindow)
    return config