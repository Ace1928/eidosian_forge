from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_volume
def flexcache_delete(self, current):
    """
        Delete FlexCache relationship at destination cluster
        Check job status
        """
    if self.parameters['force_unmount']:
        self.volume_unmount(current)
    if self.parameters['force_offline']:
        self.volume_offline(current)
    if self.use_rest:
        return self.flexcache_rest_delete(current)
    results = self.flexcache_delete_async()
    status = results.get('result-status')
    if status == 'in_progress' and 'result-jobid' in results:
        if self.parameters['time_out'] == 0:
            return None
        error = self.check_job_status(results['result-jobid'])
        if error is not None:
            self.module.fail_json(msg='Error when deleting flexcache: %s' % error)
        return None
    self.module.fail_json(msg='Unexpected error when deleting flexcache: results is: %s' % repr(results))