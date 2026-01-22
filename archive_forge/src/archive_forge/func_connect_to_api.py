from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def connect_to_api(module, disconnect_atexit=True, return_si=False, hostname=None, username=None, password=None, port=None, validate_certs=None, httpProxyHost=None, httpProxyPort=None):
    if module:
        if not hostname:
            hostname = module.params['hostname']
        if not username:
            username = module.params['username']
        if not password:
            password = module.params['password']
        if not httpProxyHost:
            httpProxyHost = module.params.get('proxy_host')
        if not httpProxyPort:
            httpProxyPort = module.params.get('proxy_port')
        if not port:
            port = module.params.get('port', 443)
        if not validate_certs:
            validate_certs = module.params['validate_certs']

    def _raise_or_fail(msg):
        if module is not None:
            module.fail_json(msg=msg)
        raise ApiAccessError(msg)
    if not hostname:
        _raise_or_fail(msg="Hostname parameter is missing. Please specify this parameter in task or export environment variable like 'export VMWARE_HOST=ESXI_HOSTNAME'")
    if not username:
        _raise_or_fail(msg="Username parameter is missing. Please specify this parameter in task or export environment variable like 'export VMWARE_USER=ESXI_USERNAME'")
    if not password:
        _raise_or_fail(msg="Password parameter is missing. Please specify this parameter in task or export environment variable like 'export VMWARE_PASSWORD=ESXI_PASSWORD'")
    if validate_certs and (not hasattr(ssl, 'SSLContext')):
        _raise_or_fail(msg='pyVim does not support changing verification mode with python < 2.7.9. Either update python or use validate_certs=false.')
    elif validate_certs:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        ssl_context.check_hostname = True
        ssl_context.load_default_certs()
    elif hasattr(ssl, 'SSLContext'):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
        ssl_context.verify_mode = ssl.CERT_NONE
        ssl_context.check_hostname = False
    else:
        ssl_context = None
    service_instance = None
    connect_args = dict(host=hostname, port=port)
    if ssl_context:
        connect_args.update(sslContext=ssl_context)
    msg_suffix = ''
    try:
        if httpProxyHost:
            msg_suffix = ' [proxy: %s:%d]' % (httpProxyHost, httpProxyPort)
            connect_args.update(httpProxyHost=httpProxyHost, httpProxyPort=httpProxyPort)
            smart_stub = connect.SmartStubAdapter(**connect_args)
            session_stub = connect.VimSessionOrientedStub(smart_stub, connect.VimSessionOrientedStub.makeUserLoginMethod(username, password))
            service_instance = vim.ServiceInstance('ServiceInstance', session_stub)
        else:
            connect_args.update(user=username, pwd=password)
            service_instance = connect.SmartConnect(**connect_args)
    except vim.fault.InvalidLogin as invalid_login:
        msg = 'Unable to log on to vCenter or ESXi API at %s:%s ' % (hostname, port)
        _raise_or_fail(msg='%s as %s: %s' % (msg, username, invalid_login.msg) + msg_suffix)
    except vim.fault.NoPermission as no_permission:
        _raise_or_fail(msg='User %s does not have required permission to log on to vCenter or ESXi API at %s:%s : %s' % (username, hostname, port, no_permission.msg))
    except (requests.ConnectionError, ssl.SSLError) as generic_req_exc:
        _raise_or_fail(msg='Unable to connect to vCenter or ESXi API at %s on TCP/%s: %s' % (hostname, port, generic_req_exc))
    except vmodl.fault.InvalidRequest as invalid_request:
        msg = 'Failed to get a response from server %s:%s ' % (hostname, port)
        _raise_or_fail(msg='%s as request is malformed: %s' % (msg, invalid_request.msg) + msg_suffix)
    except Exception as generic_exc:
        msg = 'Unknown error while connecting to vCenter or ESXi API at %s:%s' % (hostname, port) + msg_suffix
        _raise_or_fail(msg='%s : %s' % (msg, generic_exc))
    if service_instance is None:
        msg = 'Unknown error while connecting to vCenter or ESXi API at %s:%s' % (hostname, port)
        _raise_or_fail(msg=msg + msg_suffix)
    if disconnect_atexit:
        atexit.register(connect.Disconnect, service_instance)
    if return_si:
        return (service_instance, service_instance.RetrieveContent())
    return service_instance.RetrieveContent()