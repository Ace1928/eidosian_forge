import io
import socket
import sys
import threading
from http.client import UnknownProtocol, parse_headers
from http.server import SimpleHTTPRequestHandler
import breezy
from .. import (config, controldir, debug, errors, osutils, tests, trace,
from ..bzr import remote as _mod_remote
from ..transport import remote
from ..transport.http import urllib
from ..transport.http.urllib import (AbstractAuthHandler, BasicAuthHandler,
from . import features, http_server, http_utils, test_server
from .scenarios import load_tests_apply_scenarios, multiply_scenarios
def vary_by_http_activity():
    activity_scenarios = [('urllib,http', dict(_activity_server=ActivityHTTPServer, _transport=HttpTransport))]
    if features.HTTPSServerFeature.available():
        from . import ssl_certs

        class HTTPS_transport(HttpTransport):

            def __init__(self, base, _from_transport=None):
                super().__init__(base, _from_transport=_from_transport, ca_certs=ssl_certs.build_path('ca.crt'))
        activity_scenarios.append(('urllib,https', dict(_activity_server=ActivityHTTPSServer, _transport=HTTPS_transport)))
    return activity_scenarios