from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def convert_to_san_application(self, scope):
    """First convert volume to smart container using POST
           Second modify app to add new luns using PATCH
        """
    modify = dict(dummy='dummy')
    body, error = self.create_san_app_body(modify)
    self.fail_on_error(error)
    dummy, error = self.rest_app.create_application(body)
    self.fail_on_error(error)
    app_current, error = self.rest_app.get_application_uuid()
    self.fail_on_error(error)
    if app_current is None:
        self.module.fail_json(msg='Error: failed to create smart container for %s' % self.parameters['name'])
    app_modify, app_modify_warning = self.app_changes(scope)
    if app_modify_warning is not None:
        self.module.warn(app_modify_warning)
    if app_modify:
        self.modify_san_application(app_modify)