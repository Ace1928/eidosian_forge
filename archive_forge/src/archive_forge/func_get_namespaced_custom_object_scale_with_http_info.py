from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def get_namespaced_custom_object_scale_with_http_info(self, group, version, namespace, plural, name, **kwargs):
    """
        read scale of the specified namespace scoped custom object
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread =
        api.get_namespaced_custom_object_scale_with_http_info(group, version,
        namespace, plural, name, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str group: the custom resource's group (required)
        :param str version: the custom resource's version (required)
        :param str namespace: The custom resource's namespace (required)
        :param str plural: the custom resource's plural name. For TPRs this
        would be lowercase plural kind. (required)
        :param str name: the custom object's name (required)
        :return: object
                 If the method is called asynchronously,
                 returns the request thread.
        """
    all_params = ['group', 'version', 'namespace', 'plural', 'name']
    all_params.append('async_req')
    all_params.append('_return_http_data_only')
    all_params.append('_preload_content')
    all_params.append('_request_timeout')
    params = locals()
    for key, val in iteritems(params['kwargs']):
        if key not in all_params:
            raise TypeError("Got an unexpected keyword argument '%s' to method get_namespaced_custom_object_scale" % key)
        params[key] = val
    del params['kwargs']
    if 'group' not in params or params['group'] is None:
        raise ValueError('Missing the required parameter `group` when calling `get_namespaced_custom_object_scale`')
    if 'version' not in params or params['version'] is None:
        raise ValueError('Missing the required parameter `version` when calling `get_namespaced_custom_object_scale`')
    if 'namespace' not in params or params['namespace'] is None:
        raise ValueError('Missing the required parameter `namespace` when calling `get_namespaced_custom_object_scale`')
    if 'plural' not in params or params['plural'] is None:
        raise ValueError('Missing the required parameter `plural` when calling `get_namespaced_custom_object_scale`')
    if 'name' not in params or params['name'] is None:
        raise ValueError('Missing the required parameter `name` when calling `get_namespaced_custom_object_scale`')
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
    if 'name' in params:
        path_params['name'] = params['name']
    query_params = []
    header_params = {}
    form_params = []
    local_var_files = {}
    body_params = None
    header_params['Accept'] = self.api_client.select_header_accept(['application/json', 'application/yaml', 'application/vnd.kubernetes.protobuf'])
    header_params['Content-Type'] = self.api_client.select_header_content_type(['*/*'])
    auth_settings = ['BearerToken']
    return self.api_client.call_api('/apis/{group}/{version}/namespaces/{namespace}/{plural}/{name}/scale', 'GET', path_params, query_params, header_params, body=body_params, post_params=form_params, files=local_var_files, response_type='object', auth_settings=auth_settings, async_req=params.get('async_req'), _return_http_data_only=params.get('_return_http_data_only'), _preload_content=params.get('_preload_content', True), _request_timeout=params.get('_request_timeout'), collection_formats=collection_formats)