from __future__ import (absolute_import, division, print_function)
def check_for_error_and_job_results(api, response, error, rest_api, **kwargs):
    """report first error if present
       otherwise call wait_on_job and retrieve job response or error
    """
    format_error = not kwargs.pop('raw_error', False)
    if error:
        if format_error:
            error = api_error(api, error)
    elif isinstance(response, dict):
        job = None
        if 'job' in response:
            job = response['job']
        elif 'jobs' in response:
            if response['num_records'] > 1:
                error = "multiple jobs in progress, can't check status"
            else:
                job = response['jobs'][0]
        if job:
            job_response, error = rest_api.wait_on_job(job, **kwargs)
            if error:
                if format_error:
                    error = job_error(response, error)
            else:
                response['job_response'] = job_response
    return (response, error)