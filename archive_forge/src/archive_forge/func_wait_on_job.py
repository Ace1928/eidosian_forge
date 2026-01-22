from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def wait_on_job(self, job, timeout=600, increment=60):
    try:
        url = job['_links']['self']['href'].split('api/')[1]
    except Exception as err:
        self.log_error(0, 'URL Incorrect format: %s - Job: %s' % (err, job))
        return (None, 'URL Incorrect format: %s - Job: %s' % (err, job))
    error = None
    errors = []
    message = None
    runtime = 0
    retries = 0
    max_retries = 3
    done = False
    while not done:
        job_json, job_error = self.get(url, None)
        job_state = job_json.get('state', None) if job_json else None
        if job_error and job_state is None:
            errors.append(str(job_error))
            retries += 1
            if retries > max_retries:
                error = ' - '.join(errors)
                self.log_error(0, 'Job error: Reached max retries.')
                done = True
        else:
            retries = 0
            done, message, error = self._is_job_done(job_json, job_state, job_error, runtime >= timeout)
        if not done:
            time.sleep(increment)
            runtime += increment
    return (message, error)