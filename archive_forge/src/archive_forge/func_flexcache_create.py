from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_volume
def flexcache_create(self):
    """
        Create a FlexCache relationship
        Check job status
        """
    if self.use_rest:
        return self.flexcache_rest_create()
    results = self.flexcache_create_async()
    status = results.get('result-status')
    if status == 'in_progress' and 'result-jobid' in results:
        if self.parameters['time_out'] == 0:
            return
        error = self.check_job_status(results['result-jobid'])
        if error is None:
            return
        else:
            self.module.fail_json(msg='Error when creating flexcache: %s' % error)
    self.module.fail_json(msg='Unexpected error when creating flexcache: results is: %s' % repr(results))