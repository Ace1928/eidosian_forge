from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def create_namespaced_custom_object_with_http_info(self, group, version, namespace, plural, body, **kwargs):
    """
        Creates a namespace scoped Custom object
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.create_namespaced_custom_object_with_http_info(group,
        version, namespace, plural, body, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str group: The custom resource's group name (required)
        :param str version: The custom resource's version (required)
        :param str namespace: The custom resource's namespace (required)
        :param str plural: The custom resource's plural name. For TPRs this
        would be lowercase plural kind. (required)
        :param object body: The JSON schema of the Resource to create.
        (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :return: object
                 If the method is called asynchronously,
                 returns the request thread.
        """
    all_params = ['group', 'version', 'namespace', 'plural', 'body', 'pretty']
    all_params.append('async_req')
    all_params.append('_return_http_data_only')
    all_params.append('_preload_content')
    all_params.append('_request_timeout')
    params = locals()
    for key, val in iteritems(params['kwargs']):
        if key not in all_params:
            raise TypeError("Got an unexpected keyword argument '%s' to method create_namespaced_custom_object" % key)
        params[key] = val
    del params['kwargs']
    if 'group' not in params or params['group'] is None:
        raise ValueError('Missing the required parameter `group` when calling `create_namespaced_custom_object`')
    if 'version' not in params or params['version'] is None:
        raise ValueError('Missing the required parameter `version` when calling `create_namespaced_custom_object`')
    if 'namespace' not in params or params['namespace'] is None:
        raise ValueError('Missing the required parameter `namespace` when calling `create_namespaced_custom_object`')
    if 'plural' not in params or params['plural'] is None:
        raise ValueError('Missing the required parameter `plural` when calling `create_namespaced_custom_object`')
    if 'body' not in params or params['body'] is None:
        raise ValueError('Missing the required parameter `body` when calling `create_namespaced_custom_object`')
    collection_formats = {}
    path_params = {}
    if 'group' in params:
        path_params['group'] = params['group']
    if 'version' in params:
        path_params['version'] = params['version']
    if 'namespace' in params:
        path_params['namespace'] = params['namespace']
    if 'plural' in params:
        path_params['plural'] = params['plural']
    query_params = []
    if 'pretty' in params:
        query_params.append(('pretty', params['pretty']))
    header_params = {}
    form_params = []
    local_var_files = {}
    body_params = None
    if 'body' in params:
        body_params = params['body']
    header_params['Accept'] = self.api_client.select_header_accept(['application/json'])
    auth_settings = ['BearerToken']
    return self.api_client.call_api('/apis/{group}/{version}/namespaces/{namespace}/{plural}', 'POST', path_params, query_params, header_params, body=body_params, post_params=form_params, files=local_var_files, response_type='object', auth_settings=auth_settings, async_req=params.get('async_req'), _return_http_data_only=params.get('_return_http_data_only'), _preload_content=params.get('_preload_content', True), _request_timeout=params.get('_request_timeout'), collection_formats=collection_formats)