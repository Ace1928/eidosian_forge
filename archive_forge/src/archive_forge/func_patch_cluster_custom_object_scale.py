from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def patch_cluster_custom_object_scale(self, group, version, plural, name, body, **kwargs):
    """
        partially update scale of the specified cluster scoped custom object
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.patch_cluster_custom_object_scale(group, version,
        plural, name, body, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str group: the custom resource's group (required)
        :param str version: the custom resource's version (required)
        :param str plural: the custom resource's plural name. For TPRs this
        would be lowercase plural kind. (required)
        :param str name: the custom object's name (required)
        :param object body: (required)
        :return: object
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.patch_cluster_custom_object_scale_with_http_info(group, version, plural, name, body, **kwargs)
    else:
        data = self.patch_cluster_custom_object_scale_with_http_info(group, version, plural, name, body, **kwargs)
        return data