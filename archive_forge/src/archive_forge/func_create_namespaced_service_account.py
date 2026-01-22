from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def create_namespaced_service_account(self, namespace, body, **kwargs):
    """
        create a ServiceAccount
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.create_namespaced_service_account(namespace, body,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str namespace: object name and auth scope, such as for teams and
        projects (required)
        :param V1ServiceAccount body: (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :param str dry_run: When present, indicates that modifications should
        not be persisted. An invalid or unrecognized dryRun directive will
        result in an error response and no further processing of the request.
        Valid values are: - All: all dry run stages will be processed
        :param str field_manager: fieldManager is a name associated with the
        actor or entity that is making these changes. The value must be less
        than or 128 characters long, and only contain printable characters, as
        defined by https://golang.org/pkg/unicode/#IsPrint.
        :return: V1ServiceAccount
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.create_namespaced_service_account_with_http_info(namespace, body, **kwargs)
    else:
        data = self.create_namespaced_service_account_with_http_info(namespace, body, **kwargs)
        return data