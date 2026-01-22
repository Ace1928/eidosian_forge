import os
import ssl
import copy
import json
import time
import socket
import binascii
from typing import Any, Dict, Type, Union, Optional
import libcloud
from libcloud.http import LibcloudConnection, HttpLibResponseProxy
from libcloud.utils.py3 import ET, httplib, urlparse, urlencode
from libcloud.utils.misc import lowercase_keys
from libcloud.utils.retry import Retry
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.common.exceptions import exception_from_message
def async_request(self, action, params=None, data=None, headers=None, method='GET', context=None):
    """
        Perform an 'async' request to the specified path. Keep in mind that
        this function is *blocking* and 'async' in this case means that the
        hit URL only returns a job ID which is the periodically polled until
        the job has completed.

        This function works like this:

        - Perform a request to the specified path. Response should contain a
          'job_id'.

        - Returned 'job_id' is then used to construct a URL which is used for
          retrieving job status. Constructed URL is then periodically polled
          until the response indicates that the job has completed or the
          timeout of 'self.timeout' seconds has been reached.

        :type action: ``str``
        :param action: A path

        :type params: ``dict``
        :param params: Optional mapping of additional parameters to send. If
            None, leave as an empty ``dict``.

        :type data: ``unicode``
        :param data: A body of data to send with the request.

        :type headers: ``dict``
        :param headers: Extra headers to add to the request
            None, leave as an empty ``dict``.

        :type method: ``str``
        :param method: An HTTP method such as "GET" or "POST".

        :type context: ``dict``
        :param context: Context dictionary which is passed to the functions
                        which construct initial and poll URL.

        :return: An :class:`Response` instance.
        :rtype: :class:`Response` instance
        """
    request = getattr(self, self.request_method)
    kwargs = self.get_request_kwargs(action=action, params=params, data=data, headers=headers, method=method, context=context)
    response = request(**kwargs)
    kwargs = self.get_poll_request_kwargs(response=response, context=context, request_kwargs=kwargs)
    end = time.time() + self.timeout
    completed = False
    while time.time() < end and (not completed):
        response = request(**kwargs)
        completed = self.has_completed(response=response)
        if not completed:
            time.sleep(self.poll_interval)
    if not completed:
        raise LibcloudError('Job did not complete in %s seconds' % self.timeout)
    return response