from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def quote_volume(self, quote):
    response, err, on_cloud_request_id = self.rest_api.send_request('POST', '%s/volumes/quote' % self.rest_api.api_root_path, None, quote, header=self.headers)
    if err is not None:
        self.module.fail_json(changed=False, msg='Error quoting destination volume %s: %s.' % (err, response))
    wait_on_completion_api_url = '/occm/api/audit/activeTask/%s' % str(on_cloud_request_id)
    err = self.rest_api.wait_on_completion(wait_on_completion_api_url, 'volume', 'quote', 20, 5)
    if err is not None:
        self.module.fail_json(changed=False, msg=err)
    return response