from __future__ import (absolute_import, division, print_function)
import ssl
import fnmatch
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def get_jobs(module):
    jenkins_conn = get_jenkins_connection(module)
    jobs = []
    if module.params.get('name'):
        try:
            job_info = jenkins_conn.get_job_info(module.params.get('name'))
        except jenkins.NotFoundException:
            pass
        else:
            jobs.append({'name': job_info['name'], 'fullname': job_info['fullName'], 'url': job_info['url'], 'color': job_info['color']})
    else:
        all_jobs = jenkins_conn.get_all_jobs()
        if module.params.get('glob'):
            jobs.extend((j for j in all_jobs if fnmatch.fnmatch(j['fullname'], module.params.get('glob'))))
        else:
            jobs = all_jobs
        for job in jobs:
            if '_class' in job:
                del job['_class']
    if module.params.get('color'):
        jobs = [j for j in jobs if j['color'] == module.params.get('color')]
    return jobs