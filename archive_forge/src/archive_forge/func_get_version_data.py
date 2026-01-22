import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def get_version_data(session, url, authenticated=None, version_header=None):
    """Retrieve raw version data from a url.

    The return is a list of dicts of the form::

      [{
          'status': 'STABLE',
          'id': 'v2.3',
          'links': [
              {
                  'href': 'http://network.example.com/v2.3',
                  'rel': 'self',
              },
              {
                  'href': 'http://network.example.com/',
                  'rel': 'collection',
              },
          ],
          'min_version': '2.0',
          'max_version': '2.7',
       },
       ...,
      ]

    Note:
    The maximum microversion may be specified by `max_version` or `version`,
    the former superseding the latter.
    All `*version` keys are optional.
    Other keys and 'links' entries are permitted, but ignored.

    :param session: A Session object that can be used for communication.
    :type session: keystoneauth1.session.Session
    :param string url: Endpoint or discovery URL from which to retrieve data.
    :param bool authenticated: Include a token in the discovery call.
                               (optional) Defaults to None.
    :param string version_header: provide the OpenStack-API-Version header
        for services which don't return version information without it, for
        backward compatibility.
    :return: A list of dicts containing version information.
    :rtype: list(dict)
    """
    headers = {'Accept': 'application/json'}
    if version_header:
        headers['OpenStack-API-Version'] = version_header
    try:
        resp = session.get(url, headers=headers, authenticated=authenticated)
    except exceptions.Unauthorized:
        resp = session.get(url, headers=headers, authenticated=True)
    try:
        body_resp = resp.json()
    except ValueError:
        pass
    else:
        if isinstance(body_resp, list):
            raise exceptions.DiscoveryFailure('Invalid Response - List returned instead of dict')
        try:
            return body_resp['versions']['values']
        except (KeyError, TypeError):
            pass
        try:
            return body_resp['versions']
        except KeyError:
            pass
        try:
            return [body_resp['version']]
        except KeyError:
            pass
        if 'id' in body_resp:
            body_resp['status'] = Status.CURRENT
            for header in resp.headers:
                header = header.lower()
                if not header.startswith('x-openstack'):
                    continue
                if header.endswith('api-minimum-version'):
                    body_resp.setdefault('min_version', resp.headers[header])
                if header.endswith('api-maximum-version'):
                    body_resp.setdefault('version', resp.headers[header])
            return [body_resp]
    err_text = resp.text[:50] + '...' if len(resp.text) > 50 else resp.text
    raise exceptions.DiscoveryFailure('Invalid Response - Bad version data returned: %s' % err_text)