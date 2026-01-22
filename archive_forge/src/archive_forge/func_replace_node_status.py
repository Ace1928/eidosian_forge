from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def replace_node_status(self, name, body, **kwargs):
    """
        replace status of the specified Node
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.replace_node_status(name, body, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the Node (required)
        :param V1Node body: (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :param str dry_run: When present, indicates that modifications should not be persisted. An invalid or unrecognized dryRun directive will result in an error response and no further processing of the request. Valid values are: - All: all dry run stages will be processed
        :param str field_manager: fieldManager is a name associated with the actor or entity that is making these changes. The value must be less than or 128 characters long, and only contain printable characters, as defined by https://golang.org/pkg/unicode/#IsPrint.
        :return: V1Node
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.replace_node_status_with_http_info(name, body, **kwargs)
    else:
        data = self.replace_node_status_with_http_info(name, body, **kwargs)
        return data