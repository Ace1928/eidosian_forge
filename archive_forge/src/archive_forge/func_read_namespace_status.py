from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def read_namespace_status(self, name, **kwargs):
    """
        read status of the specified Namespace
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.read_namespace_status(name, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the Namespace (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :return: V1Namespace
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.read_namespace_status_with_http_info(name, **kwargs)
    else:
        data = self.read_namespace_status_with_http_info(name, **kwargs)
        return data