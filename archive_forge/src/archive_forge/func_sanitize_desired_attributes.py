from __future__ import absolute_import, division, print_function
import codecs
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_bytes, to_native, to_text
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def sanitize_desired_attributes(self):
    """ add top 'desired-attributes' if absent
            check for _ as more likely ZAPI does not take them
        """
    da_key = 'desired-attributes'
    if da_key not in self.desired_attributes:
        desired_attributes = dict()
        desired_attributes[da_key] = self.desired_attributes
        self.desired_attributes = desired_attributes
    self.check_for___in_keys(self.desired_attributes)