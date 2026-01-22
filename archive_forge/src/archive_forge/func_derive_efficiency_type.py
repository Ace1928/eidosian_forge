from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
@staticmethod
def derive_efficiency_type(desired_background, desired_inline, current_background, current_inline):
    if desired_background and desired_inline or (desired_background and desired_inline is None and current_inline) or (desired_background is None and desired_inline and current_background):
        return 'both'
    elif desired_background and desired_inline is False or (desired_background and desired_inline is None and (not current_inline)) or (desired_background is None and desired_inline is False and current_background):
        return 'background'
    elif desired_background is False and desired_inline or (desired_background is False and desired_inline is None and current_inline) or (desired_background is None and desired_inline and (not current_background)):
        return 'inline'
    elif desired_background is False and desired_inline is False or (desired_background is False and desired_inline is None and (not current_inline)) or (desired_background is None and desired_inline is False and (not current_background)):
        return 'none'