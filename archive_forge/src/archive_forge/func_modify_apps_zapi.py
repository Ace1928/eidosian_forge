from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def modify_apps_zapi(self, current, modify_decision):
    if 'applications' not in modify_decision:
        modify_decision['applications'] = self.parameters['applications']
    current_apps = dict(((application['application'], application['authentication_methods']) for application in current['applications']))
    for application in modify_decision['applications']:
        if application['application'] in current_apps:
            self.modify_user(application, current_apps[application['application']])
        else:
            self.create_user(application)
    desired_apps = dict(((application['application'], application['authentication_methods']) for application in self.parameters['applications']))
    for application in current['applications']:
        if application['application'] not in desired_apps:
            self.delete_user(application)
        else:
            self.delete_user(application, desired_apps[application['application']])