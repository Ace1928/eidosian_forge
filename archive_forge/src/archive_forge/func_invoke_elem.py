from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def invoke_elem(self, na_element, enable_tunneling=False):
    """Invoke the API on the server."""
    if not na_element or not isinstance(na_element, zapi.NaElement):
        raise ValueError('NaElement must be supplied to invoke API')
    request, request_element = self._create_request(na_element, enable_tunneling)
    if self._trace:
        zapi.LOG.debug('Request: %s', request_element.to_string(pretty=True))
    if not hasattr(self, '_opener') or not self._opener or self._refresh_conn:
        self._build_opener()
    try:
        if hasattr(self, '_timeout'):
            response = self._opener.open(request, timeout=self._timeout)
        else:
            response = self._opener.open(request)
    except zapi.urllib.error.HTTPError as exc:
        raise zapi.NaApiError(exc.code, exc.reason)
    except zapi.urllib.error.URLError as exc:
        msg = 'URL error'
        error = repr(exc)
        try:
            if isinstance(exc.reason, ConnectionRefusedError):
                msg = 'Unable to connect'
                error = exc.args
        except Exception:
            pass
        raise zapi.NaApiError(msg, error)
    except Exception as exc:
        raise zapi.NaApiError('Unexpected error', repr(exc))
    response_xml = response.read()
    response_element = self._get_result(response_xml)
    if self._trace:
        zapi.LOG.debug('Response: %s', response_element.to_string(pretty=True))
    return response_element