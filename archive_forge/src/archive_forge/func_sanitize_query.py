from __future__ import absolute_import, division, print_function
import codecs
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_bytes, to_native, to_text
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def sanitize_query(self):
    """ add top 'query' if absent
            check for _ as more likely ZAPI does not take them
        """
    key = 'query'
    if key not in self.query:
        query = dict()
        query[key] = self.query
        self.query = query
    self.check_for___in_keys(self.query)