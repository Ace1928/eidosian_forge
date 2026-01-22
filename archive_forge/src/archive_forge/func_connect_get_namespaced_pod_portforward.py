from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def connect_get_namespaced_pod_portforward(self, name, namespace, **kwargs):
    """
        connect GET requests to portforward of Pod
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.connect_get_namespaced_pod_portforward(name, namespace,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the PodPortForwardOptions (required)
        :param str namespace: object name and auth scope, such as for teams and
        projects (required)
        :param int ports: List of ports to forward Required when using
        WebSockets
        :return: str
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.connect_get_namespaced_pod_portforward_with_http_info(name, namespace, **kwargs)
    else:
        data = self.connect_get_namespaced_pod_portforward_with_http_info(name, namespace, **kwargs)
        return data