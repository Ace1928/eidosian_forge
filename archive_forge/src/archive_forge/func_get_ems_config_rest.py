from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_ems_config_rest(self):
    """Get EMS config details"""
    fields = 'mail_from,mail_server,proxy_url,proxy_user'
    if 'pubsub_enabled' in self.parameters and self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1):
        fields += ',pubsub_enabled'
    record, error = rest_generic.get_one_record(self.rest_api, 'support/ems', None, fields)
    if error:
        self.module.fail_json(msg='Error fetching EMS config: %s' % to_native(error), exception=traceback.format_exc())
    if record:
        return {'mail_from': record.get('mail_from'), 'mail_server': record.get('mail_server'), 'proxy_url': record.get('proxy_url'), 'proxy_user': record.get('proxy_user'), 'pubsub_enabled': record.get('pubsub_enabled')}
    return None