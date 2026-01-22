from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def delete_persistent_volume_with_http_info(self, name, **kwargs):
    """
        delete a PersistentVolume
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.delete_persistent_volume_with_http_info(name,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the PersistentVolume (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :param V1DeleteOptions body:
        :param str dry_run: When present, indicates that modifications should
        not be persisted. An invalid or unrecognized dryRun directive will
        result in an error response and no further processing of the request.
        Valid values are: - All: all dry run stages will be processed
        :param int grace_period_seconds: The duration in seconds before the
        object should be deleted. Value must be non-negative integer. The value
        zero indicates delete immediately. If this value is nil, the default
        grace period for the specified type will be used. Defaults to a per
        object value if not specified. zero means delete immediately.
        :param bool orphan_dependents: Deprecated: please use the
        PropagationPolicy, this field will be deprecated in 1.7. Should the
        dependent objects be orphaned. If true/false, the "orphan" finalizer
        will be added to/removed from the object's finalizers list. Either this
        field or PropagationPolicy may be set, but not both.
        :param str propagation_policy: Whether and how garbage collection will
        be performed. Either this field or OrphanDependents may be set, but not
        both. The default policy is decided by the existing finalizer set in the
        metadata.finalizers and the resource-specific default policy. Acceptable
        values are: 'Orphan' - orphan the dependents; 'Background' - allow the
        garbage collector to delete the dependents in the background;
        'Foreground' - a cascading policy that deletes all dependents in the
        foreground.
        :return: V1Status
                 If the method is called asynchronously,
                 returns the request thread.
        """
    all_params = ['name', 'pretty', 'body', 'dry_run', 'grace_period_seconds', 'orphan_dependents', 'propagation_policy']
    all_params.append('async_req')
    all_params.append('_return_http_data_only')
    all_params.append('_preload_content')
    all_params.append('_request_timeout')
    params = locals()
    for key, val in iteritems(params['kwargs']):
        if key not in all_params:
            raise TypeError("Got an unexpected keyword argument '%s' to method delete_persistent_volume" % key)
        params[key] = val
    del params['kwargs']
    if 'name' not in params or params['name'] is None:
        raise ValueError('Missing the required parameter `name` when calling `delete_persistent_volume`')
    collection_formats = {}
    path_params = {}
    if 'name' in params:
        path_params['name'] = params['name']
    query_params = []
    if 'pretty' in params:
        query_params.append(('pretty', params['pretty']))
    if 'dry_run' in params:
        query_params.append(('dryRun', params['dry_run']))
    if 'grace_period_seconds' in params:
        query_params.append(('gracePeriodSeconds', params['grace_period_seconds']))
    if 'orphan_dependents' in params:
        query_params.append(('orphanDependents', params['orphan_dependents']))
    if 'propagation_policy' in params:
        query_params.append(('propagationPolicy', params['propagation_policy']))
    header_params = {}
    form_params = []
    local_var_files = {}
    body_params = None
    if 'body' in params:
        body_params = params['body']
    header_params['Accept'] = self.api_client.select_header_accept(['application/json', 'application/yaml', 'application/vnd.kubernetes.protobuf'])
    header_params['Content-Type'] = self.api_client.select_header_content_type(['*/*'])
    auth_settings = ['BearerToken']
    return self.api_client.call_api('/api/v1/persistentvolumes/{name}', 'DELETE', path_params, query_params, header_params, body=body_params, post_params=form_params, files=local_var_files, response_type='V1Status', auth_settings=auth_settings, async_req=params.get('async_req'), _return_http_data_only=params.get('_return_http_data_only'), _preload_content=params.get('_preload_content', True), _request_timeout=params.get('_request_timeout'), collection_formats=collection_formats)