from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def poll_job(self, job=None, key=None):
    if 'jobid' in job:
        while True:
            res = self.query_api('queryAsyncJobResult', jobid=job['jobid'])
            if res['jobstatus'] != 0 and 'jobresult' in res:
                if 'errortext' in res['jobresult']:
                    self.fail_json(msg="Failed: '%s'" % res['jobresult']['errortext'])
                if key and key in res['jobresult']:
                    job = res['jobresult'][key]
                break
            time.sleep(2)
    return job