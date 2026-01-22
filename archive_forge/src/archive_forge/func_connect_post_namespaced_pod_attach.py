from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def connect_post_namespaced_pod_attach(self, name, namespace, **kwargs):
    """
        connect POST requests to attach of Pod
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.connect_post_namespaced_pod_attach(name, namespace,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the PodAttachOptions (required)
        :param str namespace: object name and auth scope, such as for teams and
        projects (required)
        :param str container: The container in which to execute the command.
        Defaults to only container if there is only one container in the pod.
        :param bool stderr: Stderr if true indicates that stderr is to be
        redirected for the attach call. Defaults to true.
        :param bool stdin: Stdin if true, redirects the standard input stream of
        the pod for this call. Defaults to false.
        :param bool stdout: Stdout if true indicates that stdout is to be
        redirected for the attach call. Defaults to true.
        :param bool tty: TTY if true indicates that a tty will be allocated for
        the attach call. This is passed through the container runtime so the tty
        is allocated on the worker node by the container runtime. Defaults to
        false.
        :return: str
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.connect_post_namespaced_pod_attach_with_http_info(name, namespace, **kwargs)
    else:
        data = self.connect_post_namespaced_pod_attach_with_http_info(name, namespace, **kwargs)
        return data