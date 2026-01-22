from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_job_schedule(self, modify, current):
    """
        modify a job schedule
        """

    def set_cron(param_key, rest_key, params, cron):
        if params[param_key] == [-1]:
            cron[rest_key] = []
        elif param_key == 'job_months' and self.month_offset == 0:
            cron[rest_key] = [x + 1 for x in params[param_key]]
        else:
            cron[rest_key] = params[param_key]
    if self.use_rest:
        cron = {}
        for param_key, rest_key in self.na_helper.params_to_rest_api_keys.items():
            if modify.get(param_key):
                set_cron(param_key, rest_key, modify, cron)
            elif current.get(param_key):
                set_cron(param_key, rest_key, current, cron)
        params = {'cron': cron}
        api = 'cluster/schedules/' + self.uuid
        dummy, error = self.rest_api.patch(api, params)
        if error is not None:
            self.module.fail_json(msg='Error modifying job schedule: %s' % error)
    else:
        job_schedule_modify = netapp_utils.zapi.NaElement.create_node_with_children('job-schedule-cron-modify', **{'job-schedule-name': self.parameters['name']})
        self.add_job_details(job_schedule_modify, modify)
        try:
            self.server.invoke_successfully(job_schedule_modify, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error modifying job schedule %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())