from __future__ import print_function
import logging
import socket
import sys
from six.moves import BaseHTTPServer
from six.moves import http_client
from six.moves import input
from six.moves import urllib
from oauth2client import _helpers
from oauth2client import client
@_helpers.positional(3)
def run_flow(flow, storage, flags=None, http=None):
    """Core code for a command-line application.

    The ``run()`` function is called from your application and runs
    through all the steps to obtain credentials. It takes a ``Flow``
    argument and attempts to open an authorization server page in the
    user's default web browser. The server asks the user to grant your
    application access to the user's data. If the user grants access,
    the ``run()`` function returns new credentials. The new credentials
    are also stored in the ``storage`` argument, which updates the file
    associated with the ``Storage`` object.

    It presumes it is run from a command-line application and supports the
    following flags:

        ``--auth_host_name`` (string, default: ``localhost``)
           Host name to use when running a local web server to handle
           redirects during OAuth authorization.

        ``--auth_host_port`` (integer, default: ``[8080, 8090]``)
           Port to use when running a local web server to handle redirects
           during OAuth authorization. Repeat this option to specify a list
           of values.

        ``--[no]auth_local_webserver`` (boolean, default: ``True``)
           Run a local web server to handle redirects during OAuth
           authorization.

    The tools module defines an ``ArgumentParser`` the already contains the
    flag definitions that ``run()`` requires. You can pass that
    ``ArgumentParser`` to your ``ArgumentParser`` constructor::

        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            parents=[tools.argparser])
        flags = parser.parse_args(argv)

    Args:
        flow: Flow, an OAuth 2.0 Flow to step through.
        storage: Storage, a ``Storage`` to store the credential in.
        flags: ``argparse.Namespace``, (Optional) The command-line flags. This
               is the object returned from calling ``parse_args()`` on
               ``argparse.ArgumentParser`` as described above. Defaults
               to ``argparser.parse_args()``.
        http: An instance of ``httplib2.Http.request`` or something that
              acts like it.

    Returns:
        Credentials, the obtained credential.
    """
    if flags is None:
        flags = argparser.parse_args()
    logging.getLogger().setLevel(getattr(logging, flags.logging_level))
    if not flags.noauth_local_webserver:
        success = False
        port_number = 0
        for port in flags.auth_host_port:
            port_number = port
            try:
                httpd = ClientRedirectServer((flags.auth_host_name, port), ClientRedirectHandler)
            except socket.error:
                pass
            else:
                success = True
                break
        flags.noauth_local_webserver = not success
        if not success:
            print(_FAILED_START_MESSAGE)
    if not flags.noauth_local_webserver:
        oauth_callback = 'http://{host}:{port}/'.format(host=flags.auth_host_name, port=port_number)
    else:
        oauth_callback = client.OOB_CALLBACK_URN
    flow.redirect_uri = oauth_callback
    authorize_url = flow.step1_get_authorize_url()
    if not flags.noauth_local_webserver:
        import webbrowser
        webbrowser.open(authorize_url, new=1, autoraise=True)
        print(_BROWSER_OPENED_MESSAGE.format(address=authorize_url))
    else:
        print(_GO_TO_LINK_MESSAGE.format(address=authorize_url))
    code = None
    if not flags.noauth_local_webserver:
        httpd.handle_request()
        if 'error' in httpd.query_params:
            sys.exit('Authentication request was rejected.')
        if 'code' in httpd.query_params:
            code = httpd.query_params['code']
        else:
            print('Failed to find "code" in the query parameters of the redirect.')
            sys.exit('Try running with --noauth_local_webserver.')
    else:
        code = input('Enter verification code: ').strip()
    try:
        credential = flow.step2_exchange(code, http=http)
    except client.FlowExchangeError as e:
        sys.exit('Authentication has failed: {0}'.format(e))
    storage.put(credential)
    credential.set_store(storage)
    print('Authentication successful.')
    return credential