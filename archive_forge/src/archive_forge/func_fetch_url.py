from __future__ import (absolute_import, division, print_function)
import atexit
import base64
import email.mime.multipart
import email.mime.nonmultipart
import email.mime.application
import email.parser
import email.utils
import functools
import io
import mimetypes
import netrc
import os
import platform
import re
import socket
import sys
import tempfile
import traceback
import types  # pylint: disable=unused-import
from contextlib import contextmanager
import ansible.module_utils.compat.typing as t
import ansible.module_utils.six.moves.http_cookiejar as cookiejar
import ansible.module_utils.six.moves.urllib.error as urllib_error
from ansible.module_utils.common.collections import Mapping, is_sequence
from ansible.module_utils.six import PY2, PY3, string_types
from ansible.module_utils.six.moves import cStringIO
from ansible.module_utils.basic import get_distribution, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
def fetch_url(module, url, data=None, headers=None, method=None, use_proxy=None, force=False, last_mod_time=None, timeout=10, use_gssapi=False, unix_socket=None, ca_path=None, cookies=None, unredirected_headers=None, decompress=True, ciphers=None, use_netrc=True):
    """Sends a request via HTTP(S) or FTP (needs the module as parameter)

    :arg module: The AnsibleModule (used to get username, password etc. (s.b.).
    :arg url:             The url to use.

    :kwarg data:          The data to be sent (in case of POST/PUT).
    :kwarg headers:       A dict with the request headers.
    :kwarg method:        "POST", "PUT", etc.
    :kwarg use_proxy:     (optional) whether or not to use proxy (Default: True)
    :kwarg boolean force: If True: Do not get a cached copy (Default: False)
    :kwarg last_mod_time: Default: None
    :kwarg int timeout:   Default: 10
    :kwarg boolean use_gssapi:   Default: False
    :kwarg unix_socket: (optional) String of file system path to unix socket file to use when establishing
        connection to the provided url
    :kwarg ca_path: (optional) String of file system path to CA cert bundle to use
    :kwarg cookies: (optional) CookieJar object to send with the request
    :kwarg unredirected_headers: (optional) A list of headers to not attach on a redirected request
    :kwarg decompress: (optional) Whether to attempt to decompress gzip content-encoded responses
    :kwarg cipher: (optional) List of ciphers to use
    :kwarg boolean use_netrc: (optional) If False: Ignores login and password in ~/.netrc file (Default: True)

    :returns: A tuple of (**response**, **info**). Use ``response.read()`` to read the data.
        The **info** contains the 'status' and other meta data. When a HttpError (status >= 400)
        occurred then ``info['body']`` contains the error response data::

    Example::

        data={...}
        resp, info = fetch_url(module,
                               "http://example.com",
                               data=module.jsonify(data),
                               headers={'Content-type': 'application/json'},
                               method="POST")
        status_code = info["status"]
        body = resp.read()
        if status_code >= 400 :
            body = info['body']
    """
    if not HAS_URLPARSE:
        module.fail_json(msg='urlparse is not installed')
    if not HAS_GZIP:
        module.fail_json(msg=GzipDecodedReader.missing_gzip_error())
    old_tempdir = tempfile.tempdir
    tempfile.tempdir = module.tmpdir
    validate_certs = module.params.get('validate_certs', True)
    if use_proxy is None:
        use_proxy = module.params.get('use_proxy', True)
    username = module.params.get('url_username', '')
    password = module.params.get('url_password', '')
    http_agent = module.params.get('http_agent', get_user_agent())
    force_basic_auth = module.params.get('force_basic_auth', '')
    follow_redirects = module.params.get('follow_redirects', 'urllib2')
    client_cert = module.params.get('client_cert')
    client_key = module.params.get('client_key')
    use_gssapi = module.params.get('use_gssapi', use_gssapi)
    if not isinstance(cookies, cookiejar.CookieJar):
        cookies = cookiejar.LWPCookieJar()
    r = None
    info = dict(url=url, status=-1)
    try:
        r = open_url(url, data=data, headers=headers, method=method, use_proxy=use_proxy, force=force, last_mod_time=last_mod_time, timeout=timeout, validate_certs=validate_certs, url_username=username, url_password=password, http_agent=http_agent, force_basic_auth=force_basic_auth, follow_redirects=follow_redirects, client_cert=client_cert, client_key=client_key, cookies=cookies, use_gssapi=use_gssapi, unix_socket=unix_socket, ca_path=ca_path, unredirected_headers=unredirected_headers, decompress=decompress, ciphers=ciphers, use_netrc=use_netrc)
        info.update(dict(((k.lower(), v) for k, v in r.info().items())))
        if PY3:
            temp_headers = {}
            for name, value in r.headers.items():
                name = name.lower()
                if name in temp_headers:
                    temp_headers[name] = ', '.join((temp_headers[name], value))
                else:
                    temp_headers[name] = value
            info.update(temp_headers)
        cookie_list = []
        cookie_dict = dict()
        for cookie in cookies:
            cookie_dict[cookie.name] = cookie.value
            cookie_list.append((cookie.name, cookie.value))
        info['cookies_string'] = '; '.join(('%s=%s' % c for c in cookie_list))
        info['cookies'] = cookie_dict
        info.update(dict(msg='OK (%s bytes)' % r.headers.get('Content-Length', 'unknown'), url=r.geturl(), status=r.code))
    except NoSSLError as e:
        distribution = get_distribution()
        if distribution is not None and distribution.lower() == 'redhat':
            module.fail_json(msg='%s. You can also install python-ssl from EPEL' % to_native(e), **info)
        else:
            module.fail_json(msg='%s' % to_native(e), **info)
    except (ConnectionError, ValueError) as e:
        module.fail_json(msg=to_native(e), **info)
    except MissingModuleError as e:
        module.fail_json(msg=to_text(e), exception=e.import_traceback)
    except urllib_error.HTTPError as e:
        r = e
        try:
            if e.fp is None:
                raise AttributeError
            body = e.read()
        except AttributeError:
            body = ''
        else:
            e.close()
        try:
            info.update(dict(((k.lower(), v) for k, v in e.info().items())))
        except Exception:
            pass
        info.update({'msg': to_native(e), 'body': body, 'status': e.code})
    except urllib_error.URLError as e:
        code = int(getattr(e, 'code', -1))
        info.update(dict(msg='Request failed: %s' % to_native(e), status=code))
    except socket.error as e:
        info.update(dict(msg='Connection failure: %s' % to_native(e), status=-1))
    except httplib.BadStatusLine as e:
        info.update(dict(msg='Connection failure: connection was closed before a valid response was received: %s' % to_native(e.line), status=-1))
    except Exception as e:
        info.update(dict(msg='An unknown error occurred: %s' % to_native(e), status=-1), exception=traceback.format_exc())
    finally:
        tempfile.tempdir = old_tempdir
    return (r, info)