from __future__ import absolute_import, division, print_function
import sys
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_user import get_users
from ansible_collections.netapp.ontap.plugins.module_utils.rest_vserver import get_vserver
def check_vserver(self, name):
    self.list_interfaces(name)
    self.list_users(vserver_name=name)