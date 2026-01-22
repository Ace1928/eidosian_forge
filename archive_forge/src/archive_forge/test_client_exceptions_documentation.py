import http.client as http_client
import eventlet.patcher
import httplib2
import webob.dec
import webob.exc
from glance.common import client
from glance.common import exception
from glance.common import wsgi
from glance.tests import functional
from glance.tests import utils

        Verify that the wsgi server does not return tracebacks to the client on
        500 errors (bug 1192132)
        