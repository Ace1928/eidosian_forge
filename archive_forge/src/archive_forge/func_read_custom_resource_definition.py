from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def read_custom_resource_definition(self, name, **kwargs):
    """
        read the specified CustomResourceDefinition
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.read_custom_resource_definition(name, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the CustomResourceDefinition (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :param bool exact: Should the export be exact.  Exact export maintains
        cluster-specific fields like 'Namespace'. Deprecated. Planned for
        removal in 1.18.
        :param bool export: Should this value be exported.  Export strips fields
        that a user can not specify. Deprecated. Planned for removal in 1.18.
        :return: V1beta1CustomResourceDefinition
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.read_custom_resource_definition_with_http_info(name, **kwargs)
    else:
        data = self.read_custom_resource_definition_with_http_info(name, **kwargs)
        return data