from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def patch_cluster_role_binding(self, name, body, **kwargs):
    """
        partially update the specified ClusterRoleBinding
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.patch_cluster_role_binding(name, body, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the ClusterRoleBinding (required)
        :param object body: (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :param str dry_run: When present, indicates that modifications should
        not be persisted. An invalid or unrecognized dryRun directive will
        result in an error response and no further processing of the request.
        Valid values are: - All: all dry run stages will be processed
        :param str field_manager: fieldManager is a name associated with the
        actor or entity that is making these changes. The value must be less
        than or 128 characters long, and only contain printable characters, as
        defined by https://golang.org/pkg/unicode/#IsPrint. This field is
        required for apply requests (application/apply-patch) but optional for
        non-apply patch types (JsonPatch, MergePatch, StrategicMergePatch).
        :param bool force: Force is going to "force" Apply requests. It means
        user will re-acquire conflicting fields owned by other people. Force
        flag must be unset for non-apply patch requests.
        :return: V1alpha1ClusterRoleBinding
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.patch_cluster_role_binding_with_http_info(name, body, **kwargs)
    else:
        data = self.patch_cluster_role_binding_with_http_info(name, body, **kwargs)
        return data