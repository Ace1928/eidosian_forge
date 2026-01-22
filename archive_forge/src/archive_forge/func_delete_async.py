from __future__ import (absolute_import, division, print_function)
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def delete_async(rest_api, api, uuid, query=None, body=None, timeout=30, job_timeout=30, headers=None, raw_error=False):
    api = '%s/%s' % (api, uuid) if uuid is not None else api
    response, error = rest_api.delete(api, body=body, params=build_query_with_timeout(query, timeout), headers=headers)
    increment = min(max(job_timeout / 6, 5), 60)
    response, error = rrh.check_for_error_and_job_results(api, response, error, rest_api, increment=increment, timeout=job_timeout, raw_error=raw_error)
    return (response, error)