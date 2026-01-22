from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def read_namespaced_replication_controller_dummy_scale(self, name, namespace, **kwargs):
    """
        read scale of the specified ReplicationControllerDummy
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread =
        api.read_namespaced_replication_controller_dummy_scale(name, namespace,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the Scale (required)
        :param str namespace: object name and auth scope, such as for teams and
        projects (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :return: ExtensionsV1beta1Scale
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.read_namespaced_replication_controller_dummy_scale_with_http_info(name, namespace, **kwargs)
    else:
        data = self.read_namespaced_replication_controller_dummy_scale_with_http_info(name, namespace, **kwargs)
        return data